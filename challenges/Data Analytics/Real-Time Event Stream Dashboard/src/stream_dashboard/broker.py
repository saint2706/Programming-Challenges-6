"""The Redis Streams side: consumer group, reads, reclaiming stale work, lag, and the atomic sink.

Delivery is at-least-once (a crashed worker's entries are redelivered). The sink turns that into
exactly-once *effect*: one Lua script checks the entry id against the ``done`` set, applies the
event's operations, adds the id and ``XACK``s, all atomically. A redelivered entry finds its id in
``done`` and only gets acknowledged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import redis
from redis.exceptions import ResponseError

from stream_dashboard.engine import Event, Op
from stream_dashboard.windows import Layout

# Only these commands may run from an operation list; SETMAX is a max-register.
APPLY_SCRIPT = """
local ALLOWED = {HINCRBY = true, SADD = true, ZADD = true, ZINCRBY = true}
local id = ARGV[1]
if redis.call('SADD', KEYS[1], id) == 0 then
  redis.call('XACK', KEYS[2], ARGV[2], id)
  return 0
end
local cur = tonumber(redis.call('GET', KEYS[3]))
local ts = tonumber(ARGV[3])
if (not cur) or ts > cur then redis.call('SET', KEYS[3], ARGV[3]) end
for _, op in ipairs(cjson.decode(ARGV[4])) do
  if not ALLOWED[op[1]] then return redis.error_reply('operation not allowed: ' .. tostring(op[1])) end
  redis.call(unpack(op))
end
redis.call('XACK', KEYS[2], ARGV[2], id)
return 1
"""


def connect(url: str) -> redis.Redis:
    return redis.Redis.from_url(url, decode_responses=True)


@dataclass(frozen=True, slots=True)
class Lag:
    stream_len: int
    undelivered: int
    pending: int

    @property
    def backlog(self) -> int:
        """Entries not yet fully handled: waiting to be read plus read but not acknowledged."""
        return self.undelivered + self.pending


class Broker:
    def __init__(self, client: redis.Redis, layout: Layout | None = None):
        self.r = client
        self.layout = layout or Layout()
        self._apply = client.register_script(APPLY_SCRIPT)

    # -- setup -----------------------------------------------------------------------------------
    def ensure_group(self) -> None:
        try:
            self.r.xgroup_create(
                self.layout.stream, self.layout.group, id="0", mkstream=True
            )
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def reset(self) -> int:
        """Delete every key under this layout's prefix. Returns how many were removed."""
        keys = list(self.r.scan_iter(match=f"{self.layout.prefix}*", count=1000))
        for i in range(0, len(keys), 500):
            self.r.delete(*keys[i : i + 500])
        return len(keys)

    # -- producing -------------------------------------------------------------------------------
    @staticmethod
    def fields(ev: Event, produced_ms: int) -> dict[str, str]:
        return {
            "ts": str(ev.ts_ms),
            "type": ev.type,
            "product": str(ev.product_id),
            "brand": ev.brand,
            "cents": str(ev.price_cents),
            "user": str(ev.user_id),
            "p": str(produced_ms),
        }

    def add_many(self, items: list[tuple[Event, int]]) -> list[str]:
        pipe = self.r.pipeline(transaction=False)
        for ev, produced_ms in items:
            pipe.xadd(self.layout.stream, self.fields(ev, produced_ms))
        return pipe.execute()

    # -- consuming -------------------------------------------------------------------------------
    def read_new(
        self, consumer: str, count: int, block_ms: int
    ) -> list[tuple[str, dict]]:
        got = self.r.xreadgroup(
            self.layout.group,
            consumer,
            {self.layout.stream: ">"},
            count=count,
            block=block_ms,
        )
        return list(got[0][1]) if got else []

    def claim_stale(
        self, consumer: str, min_idle_ms: int, count: int
    ) -> list[tuple[str, dict]]:
        """Take over entries another consumer read but never acknowledged for ``min_idle_ms``."""
        _, entries, _ = self.r.xautoclaim(
            self.layout.stream,
            self.layout.group,
            consumer,
            min_idle_time=min_idle_ms,
            start_id="0-0",
            count=count,
        )
        return [(i, f) for i, f in entries if f is not None]

    @staticmethod
    def decode(fields: dict) -> Event:
        return Event(
            ts_ms=int(fields["ts"]),
            type=fields["type"],
            product_id=int(fields["product"]),
            brand=fields["brand"],
            price_cents=int(fields["cents"]),
            user_id=int(fields["user"]),
            produced_ms=int(fields["p"]),
        )

    def max_event_ms(self) -> int | None:
        raw = self.r.get(self.layout.max_ts)
        return None if raw is None else int(raw)

    def apply_entries(self, items: list[tuple[str, int, list[Op]]]) -> list[int]:
        """Apply ``(entry_id, event_ts_ms, ops)`` atomically one entry each; 1 = applied, 0 = dup."""
        L = self.layout
        pipe = self.r.pipeline(transaction=False)
        for entry_id, ts_ms, ops in items:
            self._apply(
                keys=[L.done, L.stream, L.max_ts],
                args=[
                    entry_id,
                    L.group,
                    str(ts_ms),
                    json.dumps(ops, separators=(",", ":")),
                ],
                client=pipe,
            )
        return [int(x) for x in pipe.execute()]

    # -- observing -------------------------------------------------------------------------------
    def lag(self) -> Lag:
        stream_len = self.r.xlen(self.layout.stream)
        try:
            groups = self.r.xinfo_groups(self.layout.stream)
        except ResponseError:  # no stream yet: nothing has been produced
            return Lag(0, 0, 0)
        for g in groups:
            if g["name"] == self.layout.group:
                undelivered = g.get("lag")
                if (
                    undelivered is None
                ):  # Redis returns nil lag after deletions in the stream
                    undelivered = max(stream_len - int(g.get("entries-read") or 0), 0)
                return Lag(stream_len, int(undelivered), int(g["pending"]))
        return Lag(stream_len, stream_len, 0)

    def consumers(self) -> list[dict]:
        try:
            return self.r.xinfo_consumers(self.layout.stream, self.layout.group)
        except ResponseError:
            return []
