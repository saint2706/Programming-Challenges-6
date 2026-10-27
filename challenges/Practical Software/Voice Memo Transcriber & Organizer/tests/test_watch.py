from pathlib import Path

import pytest
from helpers import tone, write_wav
from voicememo.asr import FakeTranscriber
from voicememo.ingest import FixedEngine
from voicememo.watch import Watcher, is_candidate


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "inbox"
    root.mkdir()
    return root


@pytest.fixture
def make_watcher(library, clock):
    def build(root, *, settle=5.0, engine=None, lib=None):
        return Watcher(
            root=root,
            library=lib or library,
            engine=engine or FixedEngine(FakeTranscriber(default_text="Memo text.")),
            settle_seconds=settle,
            clock=clock,
        )

    return build


@pytest.mark.parametrize(
    "name",
    ["memo.m4a", "MEMO.WAV", "voice.mp3", "a.opus", "call.amr", "clip.3gp", "x.webm"],
)
def test_audio_names_are_candidates(name, tmp_path):
    assert is_candidate(tmp_path / name)


@pytest.mark.parametrize(
    "name",
    [
        "memo.m4a.part",
        "download.crdownload",
        "rec.tmp",
        "rec.partial",
        ".hidden.m4a",
        "~lock.m4a",
        "memo.m4a~",
        "notes.txt",
        "photo.jpg",
        "archive.m4a.swp",
    ],
)
def test_partial_hidden_and_non_audio_are_ignored(name, tmp_path):
    assert not is_candidate(tmp_path / name)


def test_new_file_waits_for_settle_window(folder, make_watcher, clock):
    write_wav(folder / "a.wav", tone(1.0))
    watcher = make_watcher(folder, settle=5.0)
    assert watcher.scan() == []
    clock.advance(4.9)
    assert watcher.scan() == []
    clock.advance(0.2)
    handled = watcher.scan()
    assert [r.status for _, r in handled] == ["added"]


def test_growing_file_restarts_the_window(folder, make_watcher, clock):
    path = folder / "a.wav"
    write_wav(path, tone(1.0))
    watcher = make_watcher(folder, settle=5.0)
    watcher.scan()
    clock.advance(4.0)
    write_wav(path, tone(2.0))  # still being copied: size changed
    watcher.scan()
    clock.advance(4.0)
    assert watcher.scan() == []  # 4 s since the last change, not 8
    clock.advance(1.5)
    assert len(watcher.scan()) == 1


def test_touch_without_size_change_restarts_window(folder, make_watcher, clock):
    import os

    path = folder / "a.wav"
    write_wav(path, tone(1.0))
    watcher = make_watcher(folder, settle=5.0)
    watcher.scan()
    clock.advance(4.0)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    watcher.scan()
    clock.advance(4.0)
    assert watcher.scan() == []


def test_each_file_is_handed_over_once(folder, make_watcher, clock):
    write_wav(folder / "a.wav", tone(1.0))
    watcher = make_watcher(folder, settle=1.0)
    watcher.scan()
    clock.advance(2.0)
    assert len(watcher.scan()) == 1
    clock.advance(2.0)
    assert watcher.scan() == []


def test_temp_files_are_never_ingested_even_when_old(folder, make_watcher, clock):
    write_wav(folder / "a.wav.part", tone(1.0))
    write_wav(folder / ".hidden.wav", tone(1.0))
    watcher = make_watcher(folder, settle=1.0)
    watcher.scan()
    clock.advance(60.0)
    assert watcher.scan() == []


def test_subdirectories_are_not_descended_into(folder, make_watcher, clock):
    (folder / "nested").mkdir()
    write_wav(folder / "nested" / "deep.wav", tone(1.0))
    watcher = make_watcher(folder, settle=1.0)
    watcher.scan()
    clock.advance(10.0)
    assert watcher.scan() == []


def test_vanished_file_restarts_settle_when_recreated(folder, make_watcher, clock):
    path = folder / "a.wav"
    write_wav(path, tone(1.0))
    watcher = make_watcher(folder, settle=5.0)
    watcher.scan()
    path.unlink()
    watcher.scan()
    write_wav(path, tone(1.0))
    clock.advance(6.0)
    assert watcher.scan() == []  # new window starts now, not at the first sighting


def test_restart_does_not_retranscribe_known_memos(
    folder, library, make_watcher, clock
):
    write_wav(folder / "a.wav", tone(1.0))
    fake = FakeTranscriber(default_text="Once only.")
    first = make_watcher(folder, settle=1.0, engine=FixedEngine(fake))
    first.scan()
    clock.advance(2.0)
    first.scan()
    assert fake.calls == ["a"]

    restarted = make_watcher(folder, settle=1.0, engine=FixedEngine(fake))
    restarted.scan()
    clock.advance(2.0)
    results = restarted.scan()
    assert [r.status for _, r in results] == ["skipped"]
    assert len(fake.calls) == 1
    assert len(library.list_memos()) == 1


def test_failed_memo_is_reported_once_and_not_retried_in_loop(
    folder, make_watcher, clock
):
    (folder / "junk.m4a").write_bytes(b"garbage" * 100)
    watcher = make_watcher(folder, settle=1.0)
    watcher.scan()
    clock.advance(2.0)
    first = watcher.scan()
    assert [r.status for _, r in first] == ["failed"]
    clock.advance(2.0)
    assert watcher.scan() == []


def test_run_scans_fixed_number_of_times_and_sleeps_between(folder, make_watcher):
    watcher = make_watcher(folder, settle=0.0)
    naps: list[float] = []
    handled = watcher.run(interval=3.0, max_scans=3, sleep=naps.append)
    assert handled == 0
    assert naps == [3.0, 3.0]


def test_run_with_single_scan_never_sleeps(folder, make_watcher):
    watcher = make_watcher(folder, settle=0.0)
    naps: list[float] = []
    watcher.run(max_scans=1, sleep=naps.append)
    assert naps == []


def test_run_counts_handled_files(folder, make_watcher, clock):
    write_wav(folder / "a.wav", tone(1.0))
    write_wav(folder / "b.wav", tone(1.0))
    watcher = make_watcher(folder, settle=0.0)
    handled = watcher.run(max_scans=2, sleep=lambda s: clock.advance(1.0))
    assert handled == 2


def test_missing_folder_raises_clearly(tmp_path, make_watcher):
    watcher = make_watcher(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        watcher.scan()


def test_unreadable_file_does_not_stop_the_scan(
    folder, make_watcher, clock, monkeypatch
):
    write_wav(folder / "a_locked.wav", tone(1.0))
    write_wav(folder / "b.wav", tone(1.0))
    real_stat = Path.stat

    def stat(self, *args, **kwargs):
        if self.name == "a_locked.wav":
            raise PermissionError("locked by another program")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    watcher = make_watcher(folder, settle=1.0)
    watcher.scan()
    clock.advance(2.0)
    assert [p.name for p, _ in watcher.scan()] == ["b.wav"]


def test_symlinks_are_never_followed(folder, tmp_path, make_watcher, clock):
    outside = write_wav(tmp_path / "outside.wav", tone(1.0))
    try:
        (folder / "link.wav").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks need extra privileges on this system")
    watcher = make_watcher(folder, settle=1.0)
    watcher.scan()
    clock.advance(2.0)
    assert watcher.scan() == []


def test_ingest_error_is_logged_and_the_file_is_not_retried(
    folder, make_watcher, clock, monkeypatch
):
    import voicememo.watch as watch_mod
    from voicememo.audio import DecodeError

    calls = []

    def refuse(*args, **kwargs):
        calls.append(1)
        raise DecodeError("too big")

    monkeypatch.setattr(watch_mod, "ingest_file", refuse)
    write_wav(folder / "huge.wav", tone(1.0))
    watcher = make_watcher(folder, settle=1.0)
    watcher.scan()
    clock.advance(2.0)
    assert watcher.scan() == []
    clock.advance(2.0)
    assert watcher.scan() == []
    assert len(calls) == 1
