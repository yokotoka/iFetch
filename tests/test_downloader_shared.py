"""
Comprehensive tests for shared-file download handling in DownloadManager.

Covers:
  - get_drive_item: owned vs shared root traversal, case-insensitivity, lazy dirs
  - _try_shared_open: all three strategies, ordering, fallthrough, edge cases
  - _open_with_retry: WSObjectNotFound fast-fail, generic 404 refresh, retryable errors
  - download_drive_item: hint field in error log for shared-file 404s
"""

import json
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ifetch.downloader import DownloadManager  # noqa: E402


# ===========================================================================
# Shared fixtures and fakes
# ===========================================================================

class _WSONotFoundError(Exception):
    """Simulates pyicloud WSObjectNotFound (cross-user shared file, 404)."""
    def __str__(self):
        return (
            'Not Found (404): {"error_code":"WSObjectNotFound",'
            '"reason":"ObjectNotFoundException: Could not find document"}'
        )


class _ObjectNotFoundError(Exception):
    """Alternate phrasing of the same Apple error (seen in older pyicloud)."""
    def __str__(self):
        return "Not Found (404): ObjectNotFoundException: Could not find document"


class _GenericNotFoundError(Exception):
    """Plain signed-URL expiry / stale cache 404 — NOT Apple's object-not-found."""
    def __str__(self):
        return "Not Found (404): signed URL expired"


class _ConnectionResetError(Exception):
    def __str__(self):
        return "Connection reset by peer"


class _FakeSharedItem(dict):
    def __getitem__(self, key):
        return super().__getitem__(key)
    def dir(self):
        return self


class _FakeDrive(dict):
    def __init__(self):
        super().__init__()
        self.shared = _FakeSharedItem()
    def __getitem__(self, key):
        return super().__getitem__(key)
    def dir(self):
        return self


class _FakePyiCloudService:
    requires_2fa = False
    requires_2sa = False

    def __init__(self, **kwargs):
        self.drive = _FakeDrive()
        self.drive["Docs"] = _FakeSharedItem()
        self.drive["Docs"]["Personal.txt"] = object()
        shared_root = _FakeSharedItem()
        shared_root["Sub"] = _FakeSharedItem()
        shared_root["Sub"]["File.txt"] = object()
        self.drive.shared["SharedRoot"] = shared_root


def _make_dm(**kwargs) -> DownloadManager:
    """Return an authenticated DownloadManager backed by _FakePyiCloudService."""
    dm = DownloadManager(email="me@example.com", **kwargs)
    dm.api = _FakePyiCloudService()
    return dm


# Shared-file item without shareID (pyicloud < 2.5 or not yet propagated)
class _SharedFileItem:
    name = "photo.jxl"
    data = {
        "docwsid": "owner-doc-uuid",
        "drivewsid": "FILE::com.apple.CloudDocs::owner-drive-uuid",
        "zone": "com.apple.CloudDocs",
        "size": 2048,
    }
    def open(self, **kwargs):
        raise _WSONotFoundError()


# Shared-file item WITH shareID (pyicloud v2.5 timlaing fork)
class _SharedFileItemWithShareID:
    name = "photo.jxl"
    data = {
        "docwsid": "owner-doc-uuid",
        "drivewsid": "FILE::com.apple.CloudDocs::owner-drive-uuid",
        "zone": "com.apple.CloudDocs",
        "size": 2048,
        "shareID": {"shareRecordName": "share-record-abc123"},
    }
    def open(self, **kwargs):
        raise _WSONotFoundError()


def _token_resp(url: str, ok: bool = True) -> mock.MagicMock:
    """Build a fake download/by_id JSON response carrying a data_token URL."""
    r = mock.MagicMock()
    r.ok = ok
    r.json.return_value = {"data_token": {"url": url}}
    return r


def _pkg_token_resp(url: str) -> mock.MagicMock:
    """Build a fake download/by_id response carrying a package_token URL."""
    r = mock.MagicMock()
    r.ok = True
    r.json.return_value = {"package_token": {"url": url}}
    return r


def _content_resp() -> mock.MagicMock:
    r = mock.MagicMock()
    r.ok = True
    return r


# ===========================================================================
# get_drive_item — traversal
# ===========================================================================

class TestGetDriveItem:
    def test_resolves_owned_path(self, monkeypatch):
        monkeypatch.setattr("ifetch.downloader.PyiCloudService", _FakePyiCloudService)
        dm = DownloadManager(email="me@example.com")
        dm.authenticate()
        assert dm.get_drive_item("Docs/Personal.txt") is not None

    def test_resolves_shared_path(self, monkeypatch):
        monkeypatch.setattr("ifetch.downloader.PyiCloudService", _FakePyiCloudService)
        dm = DownloadManager(email="me@example.com")
        dm.authenticate()
        assert dm.get_drive_item("SharedRoot/Sub/File.txt") is not None

    def test_case_insensitive_owned(self, monkeypatch):
        monkeypatch.setattr("ifetch.downloader.PyiCloudService", _FakePyiCloudService)
        dm = DownloadManager(email="me@example.com")
        dm.authenticate()
        assert dm.get_drive_item("docs/personal.txt") is not None

    def test_case_insensitive_shared(self, monkeypatch):
        monkeypatch.setattr("ifetch.downloader.PyiCloudService", _FakePyiCloudService)
        dm = DownloadManager(email="me@example.com")
        dm.authenticate()
        assert dm.get_drive_item("sharedroot/sub/file.txt") is not None

    def test_missing_path_raises(self, monkeypatch):
        monkeypatch.setattr("ifetch.downloader.PyiCloudService", _FakePyiCloudService)
        dm = DownloadManager(email="me@example.com")
        dm.authenticate()
        with pytest.raises(Exception):
            dm.get_drive_item("DoesNotExist/nowhere.txt")

    def test_lazy_root_resolved_via_dir(self, monkeypatch):
        class _LazyRoot:
            shared = None
            _listed = False
            _children = {"Documents": _FakeSharedItem()}

            def __getitem__(self, key):
                if not self._listed:
                    raise KeyError(key)
                return self._children[key]

            def dir(self):
                self._listed = True
                return list(self._children.keys())

        class _Svc:
            requires_2fa = requires_2sa = False
            def __init__(self, **kw):
                self.drive = _LazyRoot()

        monkeypatch.setattr("ifetch.downloader.PyiCloudService", _Svc)
        dm = DownloadManager(email="me@example.com")
        dm.authenticate()
        assert dm.get_drive_item("Documents") is not None

    def test_empty_path_returns_root(self, monkeypatch):
        monkeypatch.setattr("ifetch.downloader.PyiCloudService", _FakePyiCloudService)
        dm = DownloadManager(email="me@example.com")
        dm.authenticate()
        root = dm.get_drive_item("")
        assert root is dm.api.drive


# ===========================================================================
# _try_shared_open — Strategy 1: shareID in download/by_id params
# ===========================================================================

class TestTrySharedOpenStrategy1:
    """Strategy 1: manually replay download/by_id with shareID in params."""

    def _drive_with_session(self, session_get_side_effect):
        drive = mock.MagicMock()
        drive.params = {"clientId": "c1"}
        drive._document_root = "https://p33-docws.icloud.com"
        drive.session = mock.MagicMock()
        drive.session.get.side_effect = session_get_side_effect
        drive.get_file.side_effect = _WSONotFoundError()
        drive.get_node_data.return_value = {}
        return drive

    def test_succeeds_with_data_token(self):
        dm = _make_dm()
        cdn = _content_resp()
        token_r = _token_resp("https://cdn.icloud.com/a.jxl")
        dm.api.drive = self._drive_with_session([token_r, cdn])

        result = dm._try_shared_open(_SharedFileItemWithShareID())
        assert result is cdn

    def test_succeeds_with_package_token(self):
        dm = _make_dm()
        cdn = _content_resp()
        token_r = _pkg_token_resp("https://cdn.icloud.com/b.jxl")
        dm.api.drive = self._drive_with_session([token_r, cdn])

        result = dm._try_shared_open(_SharedFileItemWithShareID())
        assert result is cdn

    def test_every_session_get_carries_a_timeout(self):
        dm = _make_dm()
        token_r = _token_resp("https://cdn.icloud.com/a.jxl")
        dm.api.drive = self._drive_with_session([token_r, _content_resp()])

        dm._try_shared_open(_SharedFileItemWithShareID())

        calls = dm.api.drive.session.get.call_args_list
        assert len(calls) == 2
        assert all(c[1].get("timeout") for c in calls)

    def test_includes_share_id_and_document_id_in_params(self):
        dm = _make_dm()
        cdn = _content_resp()
        token_r = _token_resp("https://cdn.icloud.com/a.jxl")
        dm.api.drive = self._drive_with_session([token_r, cdn])

        dm._try_shared_open(_SharedFileItemWithShareID())

        first_call = dm.api.drive.session.get.call_args_list[0]
        params = first_call[1].get("params") or first_call[0][1]
        assert params["document_id"] == "owner-doc-uuid"
        assert params["shareID"] == {"shareRecordName": "share-record-abc123"}

    def test_uses_correct_zone_in_url(self):
        dm = _make_dm()
        cdn = _content_resp()
        token_r = _token_resp("https://cdn.icloud.com/a.jxl")
        dm.api.drive = self._drive_with_session([token_r, cdn])

        dm._try_shared_open(_SharedFileItemWithShareID())

        url_called = dm.api.drive.session.get.call_args_list[0][0][0]
        assert "/ws/com.apple.CloudDocs/download/by_id" in url_called

    def test_skipped_when_no_share_id(self):
        """No shareID → strategy 1 must not call drive.session at all."""
        dm = _make_dm()
        drive = mock.MagicMock()
        drive.params = {"clientId": "c1"}
        drive._document_root = "https://p33-docws.icloud.com"
        drive.session = mock.MagicMock()
        drive.get_file.side_effect = _WSONotFoundError()
        drive.get_node_data.return_value = {}
        dm.api.drive = drive

        dm._try_shared_open(_SharedFileItem())  # no shareID

        drive.session.get.assert_not_called()

    def test_falls_through_when_api_returns_non_ok(self):
        """Non-ok response from download/by_id should fall through to strategy 2."""
        dm = _make_dm()
        bad_r = mock.MagicMock()
        bad_r.ok = False
        drive = self._drive_with_session([bad_r])
        # Strategy 2 also fails; strategy 3 also fails → None
        drive.get_file.side_effect = _WSONotFoundError()
        drive.get_node_data.return_value = {}
        dm.api.drive = drive

        result = dm._try_shared_open(_SharedFileItemWithShareID())
        assert result is None

    def test_falls_through_when_neither_token_key_present(self):
        """data_token and package_token both absent → fall through."""
        dm = _make_dm()
        bad_r = mock.MagicMock()
        bad_r.ok = True
        bad_r.json.return_value = {}  # no token keys
        drive = self._drive_with_session([bad_r])
        drive.get_file.side_effect = _WSONotFoundError()
        drive.get_node_data.return_value = {}
        dm.api.drive = drive

        result = dm._try_shared_open(_SharedFileItemWithShareID())
        assert result is None

    def test_falls_through_on_exception(self):
        """Exception in strategy 1 must not propagate — fall through silently."""
        dm = _make_dm()
        drive = mock.MagicMock()
        drive.params = {"clientId": "c1"}
        drive._document_root = "https://p33-docws.icloud.com"
        drive.session = mock.MagicMock()
        drive.session.get.side_effect = ConnectionError("boom")
        drive.get_file.side_effect = _WSONotFoundError()
        drive.get_node_data.return_value = {}
        dm.api.drive = drive

        result = dm._try_shared_open(_SharedFileItemWithShareID())
        assert result is None  # didn't raise


# ===========================================================================
# _try_shared_open — Strategy 2: drivewsid as document_id
# ===========================================================================

class TestTrySharedOpenStrategy2:
    """Strategy 2: call get_file(drivewsid, zone=zone)."""

    def test_succeeds_when_drivewsid_differs(self):
        dm = _make_dm()
        fake_resp = _content_resp()
        drive = mock.MagicMock()
        # No session attrs so strategy 1 is skipped
        del drive.params
        drive.get_file.return_value = fake_resp
        drive.get_node_data.return_value = {}
        dm.api.drive = drive

        result = dm._try_shared_open(_SharedFileItem())
        assert result is fake_resp
        drive.get_file.assert_called_once_with(
            _SharedFileItem.data["drivewsid"],
            zone=_SharedFileItem.data["zone"],
            stream=True,
        )

    def test_skipped_when_drivewsid_equals_docwsid(self):
        """If drivewsid == docwsid, strategy 2 must not call get_file."""
        class _SameIDItem:
            name = "f.txt"
            data = {
                "docwsid": "same-id",
                "drivewsid": "same-id",
                "zone": "com.apple.CloudDocs",
                "size": 10,
            }
            def open(self, **kwargs):
                raise _WSONotFoundError()

        dm = _make_dm()
        drive = mock.MagicMock()
        del drive.params
        drive.get_file.return_value = _content_resp()
        drive.get_node_data.return_value = {}
        dm.api.drive = drive

        dm._try_shared_open(_SameIDItem())
        drive.get_file.assert_not_called()

    def test_falls_through_on_get_file_exception(self):
        dm = _make_dm()
        drive = mock.MagicMock()
        del drive.params
        drive.get_file.side_effect = _WSONotFoundError()
        drive.get_node_data.return_value = {}
        dm.api.drive = drive

        result = dm._try_shared_open(_SharedFileItem())
        assert result is None


# ===========================================================================
# _try_shared_open — Strategy 3: node metadata direct URL
# ===========================================================================

class TestTrySharedOpenStrategy3:
    """Strategy 3: get_node_data may return a direct download URL."""

    def _drive_no_s1_s2(self, node_data: dict) -> mock.MagicMock:
        """Drive whose strategies 1 and 2 always fail; strategy 3 returns node_data."""
        drive = mock.MagicMock()
        del drive.params  # disables strategy 1
        drive.get_file.side_effect = _WSONotFoundError()
        drive.get_node_data.return_value = node_data
        return drive

    def test_uses_downloadURL_key(self):
        dm = _make_dm()
        dm.api.drive = self._drive_no_s1_s2({"downloadURL": "https://cdn.icloud.com/f.jxl"})
        fake_resp = _content_resp()
        dm.http.get = mock.MagicMock(return_value=fake_resp)

        result = dm._try_shared_open(_SharedFileItem())
        assert result is fake_resp
        dm.http.get.assert_called_once_with("https://cdn.icloud.com/f.jxl", stream=True, timeout=30)

    def test_uses_url_key(self):
        dm = _make_dm()
        dm.api.drive = self._drive_no_s1_s2({"url": "https://cdn.icloud.com/g.jxl"})
        fake_resp = _content_resp()
        dm.http.get = mock.MagicMock(return_value=fake_resp)

        result = dm._try_shared_open(_SharedFileItem())
        assert result is fake_resp

    def test_uses_contentURL_key(self):
        dm = _make_dm()
        dm.api.drive = self._drive_no_s1_s2({"contentURL": "https://cdn.icloud.com/h.jxl"})
        fake_resp = _content_resp()
        dm.http.get = mock.MagicMock(return_value=fake_resp)

        result = dm._try_shared_open(_SharedFileItem())
        assert result is fake_resp

    def test_passes_share_id_when_pyicloud_v25_signature(self):
        """If get_node_data accepts share_id kwarg, it must be passed."""
        dm = _make_dm()
        calls = []

        def _get_node_data(drivewsid, share_id=None):
            calls.append({"drivewsid": drivewsid, "share_id": share_id})
            return {"downloadURL": "https://cdn.icloud.com/x.jxl"}

        drive = mock.MagicMock()
        del drive.params
        drive.get_file.side_effect = _WSONotFoundError()
        drive.get_node_data = _get_node_data
        dm.api.drive = drive
        fake_resp = _content_resp()
        dm.http.get = mock.MagicMock(return_value=fake_resp)

        dm._try_shared_open(_SharedFileItemWithShareID())

        assert calls[0]["share_id"] == {"shareRecordName": "share-record-abc123"}

    def test_no_share_id_passed_for_old_signature(self):
        """Older pyicloud without share_id param: must call with positional only."""
        dm = _make_dm()
        calls = []

        def _get_node_data(drivewsid):  # no share_id param
            calls.append(drivewsid)
            return {"downloadURL": "https://cdn.icloud.com/y.jxl"}

        drive = mock.MagicMock()
        del drive.params
        drive.get_file.side_effect = _WSONotFoundError()
        drive.get_node_data = _get_node_data
        dm.api.drive = drive
        dm.http.get = mock.MagicMock(return_value=_content_resp())

        dm._try_shared_open(_SharedFileItemWithShareID())  # has shareID but old pyicloud
        assert len(calls) == 1  # called without error

    def test_returns_none_when_no_url_in_node_data(self):
        dm = _make_dm()
        dm.api.drive = self._drive_no_s1_s2({"name": "photo.jxl", "size": 1024})
        result = dm._try_shared_open(_SharedFileItem())
        assert result is None

    def test_falls_through_on_get_node_data_exception(self):
        dm = _make_dm()
        drive = mock.MagicMock()
        del drive.params
        drive.get_file.side_effect = _WSONotFoundError()
        drive.get_node_data.side_effect = RuntimeError("API down")
        dm.api.drive = drive

        result = dm._try_shared_open(_SharedFileItem())
        assert result is None  # didn't raise


# ===========================================================================
# _try_shared_open — strategy ordering and full fallthrough
# ===========================================================================

class TestTrySharedOpenOrdering:
    def test_strategy1_before_strategy2(self):
        """When shareID present, strategy 1 is tried first; strategy 2 fires only if 1 fails."""
        dm = _make_dm()
        s1_called = []
        s2_called = []

        def _session_get(url, **kwargs):
            s1_called.append(url)
            r = mock.MagicMock()
            r.ok = False
            return r

        drive = mock.MagicMock()
        drive.params = {"clientId": "c1"}
        drive._document_root = "https://p33-docws.icloud.com"
        drive.session = mock.MagicMock()
        drive.session.get.side_effect = _session_get

        def _get_file(file_id, zone=None, **kw):
            s2_called.append(file_id)
            raise _WSONotFoundError()

        drive.get_file = _get_file
        drive.get_node_data.return_value = {}
        dm.api.drive = drive

        dm._try_shared_open(_SharedFileItemWithShareID())

        assert len(s1_called) == 1
        assert len(s2_called) == 1  # strategy 2 ran after strategy 1 failed

    def test_all_strategies_fail_returns_none(self):
        dm = _make_dm()
        drive = mock.MagicMock()
        drive.params = {"clientId": "c1"}
        drive._document_root = "https://p33-docws.icloud.com"
        drive.session = mock.MagicMock()
        bad_r = mock.MagicMock()
        bad_r.ok = False
        drive.session.get.return_value = bad_r
        drive.get_file.side_effect = _WSONotFoundError()
        drive.get_node_data.return_value = {}
        dm.api.drive = drive

        result = dm._try_shared_open(_SharedFileItemWithShareID())
        assert result is None

    def test_strategy2_success_skips_strategy3(self):
        """If strategy 2 succeeds, strategy 3 (get_node_data) must not be called."""
        dm = _make_dm()
        fake_resp = _content_resp()
        drive = mock.MagicMock()
        del drive.params
        drive.get_file.return_value = fake_resp
        dm.api.drive = drive

        result = dm._try_shared_open(_SharedFileItem())
        assert result is fake_resp
        drive.get_node_data.assert_not_called()


# ===========================================================================
# _try_shared_open — defensive / edge cases
# ===========================================================================

class TestTrySharedOpenEdgeCases:
    def test_returns_none_when_api_is_none(self):
        dm = _make_dm()
        dm.api = None
        assert dm._try_shared_open(_SharedFileItem()) is None

    def test_returns_none_when_drive_missing(self):
        dm = _make_dm()
        dm.api = types.SimpleNamespace()  # no 'drive' attr
        assert dm._try_shared_open(_SharedFileItem()) is None

    def test_returns_none_when_item_has_no_data_attr(self):
        dm = _make_dm()
        dm.api.drive = mock.MagicMock()
        item = types.SimpleNamespace(name="f.txt")  # no data
        assert dm._try_shared_open(item) is None

    def test_returns_none_when_item_data_is_none(self):
        dm = _make_dm()
        dm.api.drive = mock.MagicMock()
        item = types.SimpleNamespace(name="f.txt", data=None)
        assert dm._try_shared_open(item) is None

    def test_returns_none_when_docwsid_missing(self):
        dm = _make_dm()
        drive = mock.MagicMock()
        del drive.params
        drive.get_file.side_effect = _WSONotFoundError()
        drive.get_node_data.return_value = {}
        dm.api.drive = drive

        item = types.SimpleNamespace(
            name="f.txt",
            data={"drivewsid": "FILE::x::y", "zone": "com.apple.CloudDocs", "size": 10},
        )
        assert dm._try_shared_open(item) is None

    def test_uses_default_zone_when_missing(self):
        """zone defaults to com.apple.CloudDocs when absent from item.data."""
        dm = _make_dm()
        fake_resp = _content_resp()
        drive = mock.MagicMock()
        del drive.params
        drive.get_file.return_value = fake_resp
        dm.api.drive = drive

        item = types.SimpleNamespace(
            name="f.txt",
            data={"docwsid": "d1", "drivewsid": "dw1", "size": 10},  # no zone
        )
        result = dm._try_shared_open(item)
        assert result is fake_resp
        drive.get_file.assert_called_once_with("dw1", zone="com.apple.CloudDocs", stream=True)


# ===========================================================================
# _open_with_retry — WSObjectNotFound behaviour
# ===========================================================================

class TestOpenWithRetryWSObjectNotFound:
    def test_fast_fails_without_burning_retries(self):
        """WSObjectNotFound must not trigger the generic get_drive_item refresh loop."""
        dm = _make_dm(max_retries=3)
        drive = mock.MagicMock()
        drive.params = {"clientId": "c1"}
        drive._document_root = "https://p33-docws.icloud.com"
        drive.session = mock.MagicMock()
        bad_r = mock.MagicMock(); bad_r.ok = False
        drive.session.get.return_value = bad_r
        drive.get_file.side_effect = _WSONotFoundError()
        drive.get_node_data.return_value = {}
        dm.api.drive = drive

        refresh_calls = []
        dm.get_drive_item = lambda path: refresh_calls.append(path) or _SharedFileItem()

        with pytest.raises(_WSONotFoundError):
            dm._open_with_retry(_SharedFileItem())

        assert refresh_calls == [], "get_drive_item must NOT be called for WSObjectNotFound"

    def test_returns_fallback_when_try_shared_open_succeeds(self):
        """If _try_shared_open finds a route, _open_with_retry must return it."""
        dm = _make_dm()
        fake_resp = _content_resp()
        dm._try_shared_open = mock.MagicMock(return_value=fake_resp)

        result = dm._open_with_retry(_SharedFileItem())
        assert result is fake_resp

    def test_logs_shared_file_warning_when_all_strategies_fail(self, caplog):
        import logging
        dm = _make_dm()
        dm._try_shared_open = mock.MagicMock(return_value=None)

        with pytest.raises(_WSONotFoundError):
            with caplog.at_level(logging.WARNING):
                dm._open_with_retry(_SharedFileItem())

        warning_events = [
            json.loads(r.message) for r in caplog.records
            if r.levelname == "WARNING" and r.message.startswith("{")
        ]
        assert any(e["event"] == "shared_file_download_attempted" for e in warning_events)

    def test_objectnotfoundexception_also_fast_fails(self):
        """The ObjectNotFoundException variant must follow the same fast-fail path."""
        class _ObjNotFoundItem:
            name = "f.txt"
            data = _SharedFileItem.data.copy()
            def open(self, **kwargs):
                raise _ObjectNotFoundError()

        dm = _make_dm(max_retries=3)
        dm._try_shared_open = mock.MagicMock(return_value=None)
        dm.get_drive_item = mock.MagicMock()

        with pytest.raises(_ObjectNotFoundError):
            dm._open_with_retry(_ObjNotFoundItem())

        dm.get_drive_item.assert_not_called()

    def test_try_shared_open_called_once_not_per_retry(self):
        """Even with max_retries=5, _try_shared_open is called exactly once."""
        dm = _make_dm(max_retries=5)
        dm._try_shared_open = mock.MagicMock(return_value=None)

        with pytest.raises(_WSONotFoundError):
            dm._open_with_retry(_SharedFileItem())

        assert dm._try_shared_open.call_count == 1


# ===========================================================================
# _open_with_retry — generic 404 (signed-URL expiry) still refreshes
# ===========================================================================

class TestOpenWithRetryGeneric404:
    def _make_item(self, error_cls, succeed_after: int):
        """Item whose open() raises error_cls N times then returns a fake response."""
        class _ResponseCtx:
            headers = {"content-length": "0"}
            url = "https://dummy/"
            def __enter__(self): return self
            def __exit__(self, *a): return False

        class _Item:
            name = "f.txt"
            data = {}
            _calls = 0

            def open(self_, **kwargs):
                _Item._calls += 1
                if _Item._calls <= succeed_after:
                    raise error_cls()
                return _ResponseCtx()

        return _Item()

    def test_generic_404_triggers_get_drive_item_refresh(self):
        dm = _make_dm(max_retries=2)
        dm._try_shared_open = mock.MagicMock(return_value=None)

        fresh = self._make_item(_GenericNotFoundError, succeed_after=0)
        refreshed = []

        def _refresh(path):
            refreshed.append(path)
            return fresh

        dm.get_drive_item = _refresh
        stale = self._make_item(_GenericNotFoundError, succeed_after=1)
        dm._open_with_retry(stale, remote_path="Folder/f.txt")

        assert refreshed == ["Folder/f.txt"]

    def test_generic_404_without_remote_path_raises_immediately(self):
        dm = _make_dm(max_retries=3)
        dm._try_shared_open = mock.MagicMock(return_value=None)

        class _Item:
            name = "f.txt"
            data = {}
            def open(self, **kwargs):
                raise _GenericNotFoundError()

        with pytest.raises(_GenericNotFoundError):
            dm._open_with_retry(_Item())  # no remote_path

    def test_wsobjectnotfound_does_not_trigger_generic_404_path(self):
        """WSObjectNotFound must NOT fall into the generic-404 refresh branch."""
        dm = _make_dm(max_retries=2)
        dm._try_shared_open = mock.MagicMock(return_value=None)
        dm.get_drive_item = mock.MagicMock()

        with pytest.raises(_WSONotFoundError):
            dm._open_with_retry(_SharedFileItem(), remote_path="SharedFolder/photo.jxl")

        dm.get_drive_item.assert_not_called()


# ===========================================================================
# _open_with_retry — retryable connection errors
# ===========================================================================

class TestOpenWithRetryRetryable:
    def _item_fails_then_succeeds(self, error_msg: str, fail_count: int):
        class _ResponseCtx:
            headers = {"content-length": "0"}
            url = "https://dummy/"
            def __enter__(self): return self
            def __exit__(self, *a): return False

        class _Item:
            name = "f.txt"
            data = {}
            _calls = 0

            def open(self_, **kwargs):
                _Item._calls += 1
                if _Item._calls <= fail_count:
                    raise Exception(error_msg)
                return _ResponseCtx()

        return _Item()

    @pytest.mark.parametrize("error_msg", [
        "Connection reset by peer",
        "Remote end closed connection",
        "Request timeout",
        "503 Service Unavailable",
        "retry_needed: server overload",
        "internal_failure",
    ])
    def test_retryable_error_is_retried(self, error_msg):
        dm = _make_dm(max_retries=3)
        item = self._item_fails_then_succeeds(error_msg, fail_count=1)
        # Should not raise — succeeds on attempt 2
        dm._open_with_retry(item)

    def test_exhausted_retries_raises(self):
        dm = _make_dm(max_retries=2)
        class _Item:
            name = "f.txt"
            data = {}
            def open(self, **kwargs):
                raise Exception("Connection reset by peer")

        with pytest.raises(Exception, match="Connection reset"):
            dm._open_with_retry(_Item())

    def test_non_retryable_error_raises_immediately(self):
        dm = _make_dm(max_retries=3)
        calls = []

        class _Item:
            name = "f.txt"
            data = {}
            def open(self_, **kwargs):
                calls.append(1)
                raise Exception("PermissionDenied: not authorized")

        with pytest.raises(Exception, match="PermissionDenied"):
            dm._open_with_retry(_Item())

        assert len(calls) == 1  # raised on first attempt, no retries


# ===========================================================================
# download_drive_item — hint field in download_failed log
# ===========================================================================

class TestDownloadDriveItemErrorHint:
    """The download_failed log event should carry a hint when the error is
    a shared-file WSObjectNotFound, and must NOT carry one for other errors."""

    def _make_item_that_fails(self, error: Exception):
        class _Item:
            name = "f.txt"
            data = {}
            size = 0
            type = "file"
            def open(self_, **kwargs):
                raise error

        return _Item()

    def _run_and_collect_log(self, dm, item, tmp_path, caplog):
        import logging
        with caplog.at_level(logging.ERROR):
            dm.download_drive_item(item, tmp_path / "f.txt")
        return [
            json.loads(r.message) for r in caplog.records
            if r.levelname == "ERROR" and r.message.startswith("{")
        ]

    def test_wsobjectnotfound_adds_hint(self, tmp_path, caplog):
        dm = _make_dm()
        dm._try_shared_open = mock.MagicMock(return_value=None)
        events = self._run_and_collect_log(
            dm, self._make_item_that_fails(_WSONotFoundError()), tmp_path, caplog
        )
        failed = [e for e in events if e.get("event") == "download_failed"]
        assert failed, "Expected download_failed event"
        assert "hint" in failed[0], "Shared-file 404 must include hint field"

    def test_objectnotfoundexception_adds_hint(self, tmp_path, caplog):
        dm = _make_dm()
        dm._try_shared_open = mock.MagicMock(return_value=None)
        events = self._run_and_collect_log(
            dm, self._make_item_that_fails(_ObjectNotFoundError()), tmp_path, caplog
        )
        failed = [e for e in events if e.get("event") == "download_failed"]
        assert "hint" in failed[0]

    def test_generic_error_has_no_hint(self, tmp_path, caplog):
        dm = _make_dm()
        events = self._run_and_collect_log(
            dm, self._make_item_that_fails(RuntimeError("disk full")), tmp_path, caplog
        )
        failed = [e for e in events if e.get("event") == "download_failed"]
        assert failed, "Expected download_failed event"
        assert "hint" not in failed[0], "Non-shared error must not have hint field"

    def test_generic_404_has_no_hint(self, tmp_path, caplog):
        """Signed-URL expiry 404 (not WSObjectNotFound) must not add hint."""
        dm = _make_dm(max_retries=1)
        events = self._run_and_collect_log(
            dm, self._make_item_that_fails(_GenericNotFoundError()), tmp_path, caplog
        )
        failed = [e for e in events if e.get("event") == "download_failed"]
        assert failed
        assert "hint" not in failed[0]


class TestAuthenticateDefaultTimeout:
    def test_session_requests_without_timeout_get_a_default(self, monkeypatch):
        seen = []

        class _Session:
            def request(self, method, url, timeout=None, **kwargs):
                seen.append(timeout)

        class _Svc:
            requires_2fa = requires_2sa = False

            def __init__(self, **kw):
                self.session = _Session()

        monkeypatch.setattr("ifetch.downloader.PyiCloudService", _Svc)
        dm = DownloadManager(email="me@example.com")
        dm.authenticate()

        dm.api.session.request("GET", "https://x")
        dm.api.session.request("GET", "https://x", timeout=5)

        assert seen == [30, 5]
