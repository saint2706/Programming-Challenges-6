"""End to end against a Redis server: producer -> stream -> workers -> window hashes."""

from __future__ import annotations

import threading
import time

import pytest
from helpers import HUGE_LATENESS_MS, events_in_order, messy
from redis.exceptions import ResponseError
from stream_dashboard.broker import Broker
from stream_dashboard.engine import simulate
from stream_dashboard.oracle import verify
from stream_dashboard.producer import ProducerConfig, produce
from stream_dashboard.windows import DEFAULT_SPECS, Layout
from stream_dashboard.worker import SimulatedCrash, Worker, WorkerConfig


def fast(**kw) -> ProducerConfig:
    return ProducerConfig(speed=0, backpressure="none", **kw)


def worker(broker, name="w1", **kw) -> Worker:
    kw.setdefault("lateness_ms", HUGE_LATENESS_MS)
    kw.setdefault("block_ms", 50)
    return Worker(broker, WorkerConfig(name=name, **kw))


def with_arrival(events, n):
    head = events.head(n)
    return head.with_columns(arrival_ms=head["ts_ms"])


def test_single_worker_matches_polars_oracle_exactly(broker, events):
    produce(broker, messy(events, 0.1), fast())
    stats = worker(broker, batch=300).run()
    assert stats.read == stats.applied == events.height
    rep = verify(broker.r, broker.layout, events, DEFAULT_SPECS)
    assert rep.ok and rep.applied == events.height
    assert all(s.exact == s.windows_expected and s.short == 0 for s in rep.specs)
    assert broker.lag().backlog == 0


def test_redis_result_equals_the_in_memory_reference_even_with_late_drops(
    broker, events
):
    frame = messy(events, fraction=0.05, stragglers=0.02)
    produce(broker, frame, fast())
    worker(broker, lateness_ms=30_000, batch=250).run()
    ref = simulate(events_in_order(frame), DEFAULT_SPECS, 30_000, broker.layout)
    for key, fields in ref.hashes.items():
        got = {k: int(v) for k, v in broker.r.hgetall(key).items()}
        assert got == dict(fields), key
    assert sum(ref.hashes[broker.layout.late].values()) > 0  # the test really had drops


def test_late_drops_are_counted_and_tumbling_is_conserved(broker, events):
    produce(broker, messy(events, 0.05, stragglers=0.03), fast())
    worker(broker, lateness_ms=30_000).run()
    rep = verify(broker.r, broker.layout, events, DEFAULT_SPECS)
    t1 = next(s for s in rep.specs if s.spec == "t1m")
    assert t1.late > 0 and t1.short > 0 and t1.over == 0
    assert t1.events_in_windows + t1.late == events.height
    assert rep.ok


def test_redelivered_entries_are_applied_once(broker, events):
    produce(broker, with_arrival(events, 1000), fast())
    w = worker(broker, batch=1000)
    entries = broker.read_new("w1", 1000, 50)
    w.process(entries)
    pattern = f"{broker.layout.prefix}w:*:agg"
    before = {k: broker.r.hgetall(k) for k in broker.r.scan_iter(pattern)}
    w.process(entries)  # the same entries again, as if the ack reply had been lost
    assert {k: broker.r.hgetall(k) for k in before} == before
    assert w.stats.duplicates == 1000 and w.stats.applied == 1000


def test_crashed_worker_entries_are_reclaimed_with_no_loss_or_duplication(
    broker, events
):
    produce(broker, messy(events, 0.05), fast())
    doomed = worker(broker, "w1", batch=100, crash_after=300, hard_crash=False)
    with pytest.raises(SimulatedCrash):
        doomed.run()
    assert broker.lag().pending == 100  # read but never acknowledged
    assert {c["name"]: c["pending"] for c in broker.consumers()}["w1"] == 100
    stats = worker(broker, "w2", batch=100, claim_idle_ms=0).run()
    assert stats.reclaimed == 100
    assert broker.r.hget(broker.layout.stats, "reclaimed") == "100"
    rep = verify(broker.r, broker.layout, events, DEFAULT_SPECS)
    assert rep.ok and rep.applied == events.height
    assert broker.lag().backlog == 0


def test_replaying_the_whole_stream_changes_nothing(broker, events):
    produce(broker, with_arrival(events, 500), fast())
    worker(broker, "w1", batch=500).run()
    key = broker.layout.agg(DEFAULT_SPECS[0], 0)
    before = broker.r.hgetall(key)
    total = broker.r.xlen(broker.layout.stream)
    broker.r.xgroup_setid(broker.layout.stream, broker.layout.group, "0")
    again = worker(broker, "w2", batch=500).run()
    assert again.read == total and again.duplicates == total and again.applied == 0
    assert broker.r.hget(broker.layout.stats, "duplicates") == str(total)
    assert broker.r.hgetall(key) == before


def test_two_workers_share_the_load_and_agree_with_the_oracle(broker, events):
    produce(broker, messy(events, 0.05), fast())
    ws = [worker(broker, f"w{i}", batch=100, process_delay_ms=0.2) for i in (1, 2)]
    threads = [threading.Thread(target=w.run) for w in ws]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert all(w.stats.read > 0 for w in ws)
    assert sum(w.stats.applied for w in ws) == events.height
    assert verify(broker.r, broker.layout, events, DEFAULT_SPECS).ok


def test_verify_catches_a_corrupted_window(broker, events):
    produce(broker, with_arrival(events, events.height), fast())
    worker(broker).run()
    spec = DEFAULT_SPECS[0]
    start = int(broker.r.zrange(broker.layout.index(spec), 3, 3)[0])
    key = broker.layout.agg(spec, start)
    broker.r.hincrby(key, "n", 1)
    rep = verify(broker.r, broker.layout, events, DEFAULT_SPECS)
    assert not rep.ok and rep.specs[0].over == 1
    broker.r.hincrby(key, "n", -1)
    broker.r.hincrby(key, "cart", 1)
    rep = verify(broker.r, broker.layout, events, DEFAULT_SPECS)
    assert not rep.ok and rep.specs[0].wrong_other == 1


def test_verify_catches_a_lost_event(broker, events):
    produce(broker, with_arrival(events, events.height), fast())
    worker(broker).run()
    broker.r.spop(broker.layout.done)
    assert not verify(broker.r, broker.layout, events, DEFAULT_SPECS).ok


def test_sink_refuses_commands_outside_its_allow_list(broker):
    with pytest.raises(ResponseError, match="not allowed"):
        broker.apply_entries([("1-1", 1, [["DEL", broker.layout.meta]])])


def test_apply_updates_the_max_event_time_register(broker):
    broker.apply_entries([("1-1", 5_000, []), ("1-2", 3_000, []), ("1-3", 9_000, [])])
    assert broker.max_event_ms() == 9_000


def test_telemetry_reports_every_batch(broker, events):
    produce(broker, with_arrival(events, 600), fast())
    worker(broker, batch=200).run()
    rows = broker.r.xrange(broker.layout.telemetry)
    assert sum(int(f["applied"]) for _, f in rows) == 600
    assert all(float(f["lat_p95"]) >= float(f["lat_p50"]) >= 0 for _, f in rows)


# -- backpressure -------------------------------------------------------------------------------
def _run_slow_consumer(broker, frame, cfg, delay_ms=0.4):
    stop, seen = threading.Event(), []

    def watch():
        while not stop.is_set():
            seen.append(broker.lag().backlog)
            time.sleep(0.005)

    watcher = threading.Thread(target=watch)
    watcher.start()
    w = worker(broker, batch=50, process_delay_ms=delay_ms)
    wt = threading.Thread(target=w.run)
    wt.start()
    res = produce(broker, frame, cfg)
    wt.join(timeout=60)
    stop.set()
    watcher.join()
    return res, seen, w


def test_backpressure_block_bounds_the_backlog_and_loses_nothing(broker, events):
    cfg = ProducerConfig(
        speed=0, batch=100, backpressure="block", high_water=300, low_water=100
    )
    res, seen, w = _run_slow_consumer(broker, with_arrival(events, 3000), cfg)
    assert max(seen) <= 300 + cfg.batch
    assert res.blocked_ms > 0
    assert w.stats.applied == 3000


def test_without_backpressure_the_backlog_grows(broker, events):
    cfg = ProducerConfig(speed=0, batch=100, backpressure="none")
    res, seen, w = _run_slow_consumer(broker, with_arrival(events, 3000), cfg)
    assert max(seen) > 1500 and res.blocked_ms == 0
    assert w.stats.applied == 3000  # nothing lost, only late


def test_producer_meta_and_replay_clock(broker, events):
    frame = with_arrival(events, 2000)
    span_s = (frame["ts_ms"].max() - frame["ts_ms"].min()) / 1000
    t0 = time.monotonic()
    produce(broker, frame, ProducerConfig(speed=span_s / 0.5, backpressure="none"))
    assert (
        0.4 < time.monotonic() - t0 < 3.0
    )  # the whole span replays in about half a second
    meta = broker.r.hgetall(broker.layout.meta)
    assert meta["done"] == "1" and int(meta["sent"]) == 2000 == int(meta["total"])


def test_producer_stop_flag(broker, events):
    calls = iter(range(10**6))
    res = produce(
        broker,
        with_arrival(events, 2000),
        fast(batch=100),
        stop=lambda: next(calls) >= 3,
    )
    assert 0 < res.sent < 2000
    assert broker.r.hget(broker.layout.meta, "done") == "1"


def test_producer_config_validation():
    with pytest.raises(ValueError):
        ProducerConfig(backpressure="maybe")
    with pytest.raises(ValueError):
        ProducerConfig(low_water=10, high_water=5)
    with pytest.raises(ValueError):
        ProducerConfig(speed=-1)


def test_reset_clears_only_its_own_prefix(broker):
    other = Broker(broker.r, Layout(prefix="sdtest:other:"))
    other.r.set("sdtest:other:k", "1")
    broker.r.set(f"{broker.layout.prefix}k", "1")
    assert broker.reset() >= 2
    assert other.r.get("sdtest:other:k") == "1"
    other.reset()
