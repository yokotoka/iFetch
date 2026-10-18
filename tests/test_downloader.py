import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ifetch.downloader import MAX_RANGE_BYTES, DownloadManager  # noqa: E402


class _FakeResp:
    def __init__(self, content: bytes, status_code: int = 206):
        self.content = content
        self.status_code = status_code
        self.url = "http://example.com"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("HTTP error")


def test_calculate_checksum(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("hello")

    dm = DownloadManager(email="user@example.com")
    checksum = dm.calculate_checksum(file_path)

    import hashlib
    expected = hashlib.sha256(b"hello").hexdigest()
    assert checksum == expected


def test_download_chunk_success(monkeypatch):
    dm = DownloadManager(email="user@example.com", max_retries=1)

    def _fake_get(url, headers, stream, timeout):
        return _FakeResp(b"abc", status_code=206)

    monkeypatch.setattr(dm.http, "get", _fake_get)

    data = dm.download_chunk("http://example.com/file", 0, 2)
    assert data == b"abc"


def test_merge_ranges_coalesces_adjacent_segments():
    dm = DownloadManager(email="user@example.com")

    merged = dm._merge_ranges([(5, 9), (0, 4), (12, 14), (10, 11)])

    assert merged == [(0, 14)]


def test_merge_ranges_bounds_range_length():
    dm = DownloadManager(email="user@example.com")
    mib = 1024 * 1024
    ranges = [(i * mib, (i + 1) * mib - 1) for i in range(300)]

    merged = dm._merge_ranges(ranges)

    assert all(end - start + 1 <= MAX_RANGE_BYTES for start, end in merged)
    assert all(cur[0] == prev[1] + 1 for prev, cur in zip(merged, merged[1:]))
    assert merged[0][0] == 0
    assert merged[-1][1] == 300 * mib - 1
