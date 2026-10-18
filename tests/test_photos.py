"""Tests for the iCloud Photos engine (ifetch.photos).

Everything here runs against fakes that mimic the two pyicloud generations:

* pyicloud 2.0.x: ``albums`` iterates album *names*, ``PhotoAsset.download``
  returns a streaming ``requests.Response``-like object.
* pyicloud >= 2.5: ``albums`` iterates album *objects* and exposes
  ``find(name)``, ``PhotoAsset.download`` returns ``bytes``.

No network access is performed.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ifetch.photos import (  # noqa: E402
    INDEX_FILENAME,
    REPORT_FILENAME,
    PhotosDownloader,
    PhotosIndex,
    parse_date,
    sanitize_filename,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeResponse:
    """Mimics the streaming Response returned by pyicloud 2.0.x."""

    def __init__(self, data: bytes, status_code: int = 200):
        self._data = data
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1024):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i:i + chunk_size]


class FakeAsset:
    """A stand-in for pyicloud's PhotoAsset."""

    def __init__(
        self,
        asset_id,
        filename,
        created,
        data=b"photo-bytes",
        live_data=None,
        live_key="original_video",
        raise_times=0,
        error=None,
        legacy_live=False,
        edited_data=None,
        edited_type="public.jpeg",
        original_type="public.jpeg",
    ):
        self.id = asset_id
        self.filename = filename
        self.created = created
        self.asset_date = created
        self._data = data
        self.size = len(data)
        self.download_calls = []
        self._raise_times = raise_times
        self._error = error or RuntimeError("transient failure")
        self.urls = {}
        self.versions = {
            "original": {
                "filename": filename,
                "url": f"https://example.invalid/{asset_id}/original",
                "size": len(data),
                "type": original_type,
            },
            "medium": {
                "filename": filename,
                "url": f"https://example.invalid/{asset_id}/medium",
                "size": 4,
                "type": "public.jpeg",
            },
        }
        self._live_data = live_data
        for info in self.versions.values():
            self.urls[info["url"]] = data
        if live_data is not None and not legacy_live:
            self.versions[live_key] = {
                "filename": Path(filename).stem + ".MOV",
                "url": f"https://example.invalid/{asset_id}/live",
                "size": len(live_data),
                "type": "com.apple.quicktime-movie",
            }
            self.urls[self.versions[live_key]["url"]] = live_data
        if live_data is not None and legacy_live:
            # pyicloud 2.0.x shape: the Live Photo video only exists as a raw
            # CloudKit field on the master record.
            self.urls[f"https://example.invalid/{asset_id}/legacy-live"] = live_data
            self._master_record = {
                "fields": {
                    "resOriginalVidComplRes": {
                        "value": {
                            "downloadURL": f"https://example.invalid/{asset_id}/legacy-live",
                            "size": len(live_data),
                        }
                    }
                }
            }

        if edited_data is not None:
            prefix = "resVidFull" if original_type == "com.apple.quicktime-movie" else "resJPEGFull"
            url = f"https://example.invalid/{asset_id}/edited"
            self.urls[url] = edited_data
            self.asset_record = {
                "fields": {
                    prefix + "Res": {"value": {"downloadURL": url, "size": len(edited_data)}},
                    prefix + "FileType": {"value": edited_type},
                }
            }

    def download_url(self, version="original"):
        self.download_calls.append(version)
        if self._raise_times > 0:
            self._raise_times -= 1
            raise self._error
        return self.versions[version]["url"]


class FakeAlbum:
    def __init__(self, name, assets, fullname=None):
        self.name = name
        self.fullname = fullname or name
        self._assets = list(assets)

    @property
    def photos(self):
        return iter(self._assets)

    def __iter__(self):
        return iter(self._assets)

    def __len__(self):
        return len(self._assets)


class ModernAlbumContainer:
    """pyicloud >= 2.5: iterates album objects, offers find()."""

    def __init__(self, albums):
        self._albums = list(albums)

    def __iter__(self):
        return iter(self._albums)

    def __len__(self):
        return len(self._albums)

    def find(self, name):
        for album in self._albums:
            if album.name == name or album.fullname == name:
                return album
        return None

    def __getitem__(self, key):
        found = self.find(key)
        if found is None:
            raise KeyError(key)
        return found


class LegacyAlbumContainer:
    """pyicloud 2.0.x: iterates album names, indexed by name."""

    def __init__(self, albums):
        self._albums = {album.name: album for album in albums}

    def __iter__(self):
        return iter(self._albums)

    def __len__(self):
        return len(self._albums)

    def __getitem__(self, key):
        return self._albums[key]


class FakeSession:
    """Serves each asset's URLs and records every GET call."""

    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.payloads[url])


class FakePhotosService:
    def __init__(self, albums, container_cls=ModernAlbumContainer, all_album=None):
        self.albums = container_cls(albums)
        self._all = all_album if all_album is not None else albums[0]
        self.session = FakeSession(
            {u: d for a in albums for x in a._assets for u, d in x.urls.items()}
        )

    @property
    def all(self):
        return self._all


class FakeAPI:
    def __init__(self, photos):
        self.photos = photos


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def make_downloader(api, tmp_path=None, monkeypatch=None, **kwargs):
    kwargs.setdefault("email", "user@example.com")
    kwargs.setdefault("max_retries", 2)
    dl = PhotosDownloader(**kwargs)
    dl.api = api
    return dl


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("ifetch.photos.time.sleep", lambda *_a, **_k: None)


def dt(year, month, day):
    return datetime(year, month, day, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def test_parse_date_accepts_common_formats():
    assert parse_date("2024-05-06") == datetime(2024, 5, 6, tzinfo=timezone.utc)
    assert parse_date("2024-05-06T07:08:09") == datetime(
        2024, 5, 6, 7, 8, 9, tzinfo=timezone.utc
    )
    with pytest.raises(ValueError):
        parse_date("not-a-date")


def test_sanitize_filename_strips_path_separators():
    assert sanitize_filename("a/b:c.jpg") == "a_b_c.jpg"
    assert sanitize_filename("") == "photo"


def test_invalid_folder_structure_rejected():
    with pytest.raises(ValueError):
        PhotosDownloader(email="user@example.com", folder_structure="bogus")


# ---------------------------------------------------------------------------
# Album listing
# ---------------------------------------------------------------------------
def test_list_albums_modern_container():
    api = FakeAPI(
        FakePhotosService(
            [
                FakeAlbum("Library", [FakeAsset("a1", "one.jpg", dt(2024, 1, 2))]),
                FakeAlbum("Family", []),
            ]
        )
    )
    dl = make_downloader(api)

    albums = dl.list_albums()

    assert [a["fullname"] for a in albums] == ["Library", "Family"]
    assert albums[0]["count"] == 1
    assert albums[1]["count"] == 0


def test_list_albums_legacy_container_of_names():
    api = FakeAPI(
        FakePhotosService(
            [FakeAlbum("Library", []), FakeAlbum("Trip", [])],
            container_cls=LegacyAlbumContainer,
        )
    )
    dl = make_downloader(api)

    names = [a["fullname"] for a in dl.list_albums()]

    assert names == ["Library", "Trip"]


def test_get_album_by_name_and_missing_album():
    family = FakeAlbum("Family", [])
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", []), family]))
    dl = make_downloader(api)

    assert dl.get_album("Family") is family
    assert dl.get_album(None) is not None

    with pytest.raises(ValueError) as exc:
        dl.get_album("Nope")
    assert "Album not found" in str(exc.value)


def test_get_album_is_case_insensitive_on_legacy_container():
    trip = FakeAlbum("Trip", [])
    api = FakeAPI(
        FakePhotosService([FakeAlbum("Library", []), trip],
                          container_cls=LegacyAlbumContainer)
    )
    dl = make_downloader(api)

    assert dl.get_album("trip") is trip


# ---------------------------------------------------------------------------
# Asset iteration + date filtering
# ---------------------------------------------------------------------------
def test_iter_assets_yields_every_asset():
    assets = [
        FakeAsset("a1", "one.jpg", dt(2024, 1, 2)),
        FakeAsset("a2", "two.jpg", dt(2024, 3, 4)),
    ]
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", assets)]))
    dl = make_downloader(api)

    got = [asset.id for _, asset in dl.iter_assets()]

    assert got == ["a1", "a2"]


def test_date_range_filtering():
    assets = [
        FakeAsset("old", "old.jpg", dt(2020, 1, 1)),
        FakeAsset("mid", "mid.jpg", dt(2023, 6, 15)),
        FakeAsset("new", "new.jpg", dt(2025, 12, 31)),
    ]
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", assets)]))
    dl = make_downloader(api)

    selected = [
        asset.id
        for _, asset in dl.iter_assets(since=dt(2023, 1, 1), until=dt(2024, 1, 1))
    ]

    assert selected == ["mid"]


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------
def test_download_originals_flat(tmp_path):
    asset = FakeAsset("a1", "one.jpg", dt(2024, 1, 2), data=b"aaaa")
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", [asset])]))
    dl = make_downloader(api, folder_structure="flat")

    report = dl.download(destination=tmp_path)

    assert (tmp_path / "one.jpg").read_bytes() == b"aaaa"
    assert asset.download_calls == ["original"]
    assert report["summary"]["downloaded"] == 1
    assert report["summary"]["failed"] == 0
    assert (tmp_path / REPORT_FILENAME).exists()


def test_date_based_folder_structure(tmp_path):
    assets = [
        FakeAsset("a1", "one.jpg", dt(2024, 1, 2)),
        FakeAsset("a2", "two.jpg", dt(2023, 11, 30)),
    ]
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", assets)]))
    dl = make_downloader(api, folder_structure="date")

    dl.download(destination=tmp_path)

    assert (tmp_path / "2024" / "01" / "one.jpg").exists()
    assert (tmp_path / "2023" / "11" / "two.jpg").exists()


def test_day_and_album_folder_structures(tmp_path):
    asset = FakeAsset("a1", "one.jpg", dt(2024, 1, 2))
    api = FakeAPI(FakePhotosService([FakeAlbum("Trip", [asset])]))

    day = make_downloader(api, folder_structure="day")
    day.download(destination=tmp_path / "day")
    assert (tmp_path / "day" / "2024" / "01" / "02" / "one.jpg").exists()

    asset2 = FakeAsset("a1", "one.jpg", dt(2024, 1, 2))
    api2 = FakeAPI(FakePhotosService([FakeAlbum("Trip", [asset2])]))
    by_album = make_downloader(api2, folder_structure="album")
    by_album.download(destination=tmp_path / "album")
    assert (tmp_path / "album" / "Trip" / "one.jpg").exists()


def test_mtime_set_to_capture_date(tmp_path):
    created = dt(2024, 1, 2)
    asset = FakeAsset("a1", "one.jpg", created)
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", [asset])]))
    dl = make_downloader(api, folder_structure="flat")

    dl.download(destination=tmp_path)

    mtime = (tmp_path / "one.jpg").stat().st_mtime
    assert abs(mtime - created.timestamp()) < 2


def test_medium_version_selection(tmp_path):
    asset = FakeAsset("a1", "one.jpg", dt(2024, 1, 2))
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", [asset])]))
    dl = make_downloader(api, folder_structure="flat", version="medium")

    dl.download(destination=tmp_path)

    assert asset.download_calls == ["medium"]


def test_unavailable_version_falls_back_to_original(tmp_path):
    asset = FakeAsset("a1", "one.jpg", dt(2024, 1, 2))
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", [asset])]))
    dl = make_downloader(api, folder_structure="flat", version="thumb")

    dl.download(destination=tmp_path)

    assert asset.download_calls == ["original"]


# ---------------------------------------------------------------------------
# Index / skip-already-downloaded
# ---------------------------------------------------------------------------
def test_second_run_skips_already_downloaded(tmp_path):
    def build():
        asset = FakeAsset("a1", "one.jpg", dt(2024, 1, 2), data=b"aaaa")
        return asset, FakeAPI(FakePhotosService([FakeAlbum("Library", [asset])]))

    asset1, api1 = build()
    first = make_downloader(api1, folder_structure="flat")
    report1 = first.download(destination=tmp_path)
    assert report1["summary"]["downloaded"] == 1
    assert (tmp_path / INDEX_FILENAME).exists()

    asset2, api2 = build()
    second = make_downloader(api2, folder_structure="flat")
    report2 = second.download(destination=tmp_path)

    assert asset2.download_calls == []
    assert report2["summary"]["downloaded"] == 0
    assert report2["summary"]["skipped"] == 1


def test_missing_local_file_is_redownloaded(tmp_path):
    asset = FakeAsset("a1", "one.jpg", dt(2024, 1, 2))
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", [asset])]))
    make_downloader(api, folder_structure="flat").download(destination=tmp_path)

    (tmp_path / "one.jpg").unlink()

    asset2 = FakeAsset("a1", "one.jpg", dt(2024, 1, 2))
    api2 = FakeAPI(FakePhotosService([FakeAlbum("Library", [asset2])]))
    report = make_downloader(api2, folder_structure="flat").download(destination=tmp_path)

    assert asset2.download_calls == ["original"]
    assert report["summary"]["downloaded"] == 1


def test_index_records_asset_metadata(tmp_path):
    asset = FakeAsset("a1", "one.jpg", dt(2024, 1, 2), data=b"abcd")
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", [asset])]))
    make_downloader(api, folder_structure="flat").download(destination=tmp_path)

    data = json.loads((tmp_path / INDEX_FILENAME).read_text())

    entry = data["assets"]["a1"]
    assert entry["filename"] == "one.jpg"
    assert entry["path"] == "one.jpg"
    assert entry["size"] == 4
    assert entry["version"] == "original"
    assert entry["created"].startswith("2024-01-02")


def test_index_tolerates_corrupt_file(tmp_path):
    (tmp_path / INDEX_FILENAME).write_text("{not json")

    index = PhotosIndex.load(tmp_path)

    assert index.entries == {}


# ---------------------------------------------------------------------------
# Filename collisions
# ---------------------------------------------------------------------------
def test_filename_collision_disambiguated(tmp_path):
    assets = [
        FakeAsset("asset0001", "IMG_0001.JPG", dt(2024, 1, 2), data=b"first"),
        FakeAsset("asset0002", "IMG_0001.JPG", dt(2024, 1, 3), data=b"second"),
    ]
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", assets)]))
    dl = make_downloader(api, folder_structure="flat")

    report = dl.download(destination=tmp_path)

    files = sorted(p.name for p in tmp_path.iterdir() if p.suffix.upper() == ".JPG")
    assert files == ["IMG_0001.JPG", "IMG_0001_sset0002.JPG"]
    assert report["summary"]["downloaded"] == 2
    # Both payloads survived: nothing was silently overwritten.
    contents = {(tmp_path / name).read_bytes() for name in files}
    assert contents == {b"first", b"second"}


def test_collision_suffix_is_deterministic(tmp_path):
    def run(target):
        assets = [
            FakeAsset("asset0001", "IMG_0001.JPG", dt(2024, 1, 2), data=b"first"),
            FakeAsset("asset0002", "IMG_0001.JPG", dt(2024, 1, 3), data=b"second"),
        ]
        api = FakeAPI(FakePhotosService([FakeAlbum("Library", assets)]))
        make_downloader(api, folder_structure="flat").download(destination=target)
        return sorted(p.name for p in target.iterdir() if p.suffix.upper() == ".JPG")

    assert run(tmp_path / "a") == run(tmp_path / "b")


def test_preexisting_unrelated_file_is_not_overwritten(tmp_path):
    (tmp_path / "one.jpg").write_bytes(b"pre-existing")
    asset = FakeAsset("a1", "one.jpg", dt(2024, 1, 2), data=b"new")
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", [asset])]))

    make_downloader(api, folder_structure="flat").download(destination=tmp_path)

    assert (tmp_path / "one.jpg").read_bytes() == b"pre-existing"
    assert any(p.name.startswith("one_") for p in tmp_path.iterdir())


# ---------------------------------------------------------------------------
# Live Photos
# ---------------------------------------------------------------------------
def test_live_photo_video_downloaded_when_flag_set(tmp_path):
    asset = FakeAsset(
        "a1", "one.jpg", dt(2024, 1, 2), data=b"still", live_data=b"movie"
    )
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", [asset])]))
    dl = make_downloader(api, folder_structure="flat", live_photos=True)

    dl.download(destination=tmp_path)

    assert (tmp_path / "one.jpg").read_bytes() == b"still"
    assert (tmp_path / "one.MOV").read_bytes() == b"movie"
    assert asset.download_calls == ["original", "original_video"]


def test_live_photo_not_fetched_without_flag(tmp_path):
    asset = FakeAsset(
        "a1", "one.jpg", dt(2024, 1, 2), data=b"still", live_data=b"movie"
    )
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", [asset])]))
    dl = make_downloader(api, folder_structure="flat", live_photos=False)

    dl.download(destination=tmp_path)

    assert not (tmp_path / "one.MOV").exists()
    assert asset.download_calls == ["original"]


def test_live_photo_legacy_cloudkit_field_fallback(tmp_path):
    """pyicloud 2.0.x has no live version key; the raw field is used instead."""
    asset = FakeAsset(
        "a1",
        "one.jpg",
        dt(2024, 1, 2),
        data=b"still",
        live_data=b"legacy-movie",
        legacy_live=True,
    )
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", [asset])]))
    dl = make_downloader(api, folder_structure="flat", live_photos=True)

    requested = {}

    class _Session:
        def get(self, url, **kwargs):
            requested["url"] = url
            return FakeResponse(b"legacy-movie")

    dl.photos_service.session = _Session()

    dl.download(destination=tmp_path)

    assert requested["url"].endswith("/legacy-live")
    assert (tmp_path / "one.MOV").read_bytes() == b"legacy-movie"


def test_live_photo_index_skips_on_rerun(tmp_path):
    def build():
        asset = FakeAsset(
            "a1", "one.jpg", dt(2024, 1, 2), data=b"still", live_data=b"movie"
        )
        return asset, FakeAPI(FakePhotosService([FakeAlbum("Library", [asset])]))

    asset1, api1 = build()
    make_downloader(api1, folder_structure="flat", live_photos=True).download(
        destination=tmp_path
    )

    asset2, api2 = build()
    report = make_downloader(api2, folder_structure="flat", live_photos=True).download(
        destination=tmp_path
    )

    assert asset2.download_calls == []
    assert report["summary"]["skipped"] == 1


def test_edited_render_downloaded_next_to_original(tmp_path):
    photo = FakeAsset("a1", "IMG_0001.HEIC", dt(2024, 1, 2), data=b"orig", edited_data=b"edit")
    video = FakeAsset(
        "a2", "MOV_0002.MOV", dt(2024, 1, 2), data=b"orig", edited_data=b"cut",
        edited_type="com.apple.quicktime-movie", original_type="com.apple.quicktime-movie",
    )
    plain = FakeAsset("a3", "IMG_0003.HEIC", dt(2024, 1, 2), data=b"orig")
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", [photo, video, plain])]))

    make_downloader(api, folder_structure="flat").download(destination=tmp_path)

    assert (tmp_path / "IMG_0001_edited.JPG").read_bytes() == b"edit"
    assert (tmp_path / "MOV_0002_edited.MOV").read_bytes() == b"cut"
    assert sorted(p.name for p in tmp_path.glob("IMG_0003*")) == ["IMG_0003.HEIC"]


def test_edited_render_fetched_for_already_downloaded_asset(tmp_path):
    def build(edited_data):
        asset = FakeAsset("a1", "one.jpg", dt(2024, 1, 2), data=b"orig", edited_data=edited_data)
        return asset, FakeAPI(FakePhotosService([FakeAlbum("Library", [asset])]))

    _, api1 = build(None)
    make_downloader(api1, folder_structure="flat").download(destination=tmp_path)
    asset2, api2 = build(b"edit")
    report = make_downloader(api2, folder_structure="flat").download(destination=tmp_path)

    assert asset2.download_calls == []
    assert (tmp_path / "one_edited.JPG").read_bytes() == b"edit"
    assert report["summary"]["downloaded"] == 1
    assert json.loads((tmp_path / INDEX_FILENAME).read_text())["assets"]["a1"]["edited_path"] == "one_edited.JPG"

    asset3, api3 = build(b"edit")
    again = make_downloader(api3, folder_structure="flat").download(destination=tmp_path)
    assert again["summary"]["skipped"] == 1


# ---------------------------------------------------------------------------
# Retry / failure containment
# ---------------------------------------------------------------------------
def test_retry_on_transient_error(tmp_path):
    asset = FakeAsset("a1", "one.jpg", dt(2024, 1, 2), data=b"ok", raise_times=1)
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", [asset])]))
    dl = make_downloader(api, folder_structure="flat", max_retries=3)

    report = dl.download(destination=tmp_path)

    assert asset.download_calls == ["original", "original"]
    assert (tmp_path / "one.jpg").read_bytes() == b"ok"
    assert report["summary"]["downloaded"] == 1


def test_retry_honours_retry_after_header(monkeypatch, tmp_path):
    class _Resp:
        headers = {"Retry-After": "7"}

    error = RuntimeError("rate limited")
    error.response = _Resp()

    asset = FakeAsset("a1", "one.jpg", dt(2024, 1, 2), raise_times=1, error=error)
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", [asset])]))
    dl = make_downloader(api, folder_structure="flat", max_retries=3)

    slept = []
    monkeypatch.setattr("ifetch.photos.time.sleep", lambda s: slept.append(s))

    dl.download(destination=tmp_path)

    assert slept == [7]


def test_non_retryable_error_is_not_retried(tmp_path):
    asset = FakeAsset(
        "a1",
        "one.jpg",
        dt(2024, 1, 2),
        raise_times=99,
        error=RuntimeError("403 Forbidden"),
    )
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", [asset])]))
    dl = make_downloader(api, folder_structure="flat", max_retries=5)

    report = dl.download(destination=tmp_path)

    assert len(asset.download_calls) == 1
    assert report["summary"]["failed"] == 1


def test_one_bad_asset_does_not_abort_the_run(tmp_path):
    good_a = FakeAsset("a1", "one.jpg", dt(2024, 1, 2), data=b"one")
    bad = FakeAsset("a2", "two.jpg", dt(2024, 1, 3), raise_times=99)
    good_b = FakeAsset("a3", "three.jpg", dt(2024, 1, 4), data=b"three")
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", [good_a, bad, good_b])]))
    dl = make_downloader(api, folder_structure="flat", max_retries=1, max_workers=2)

    report = dl.download(destination=tmp_path)

    assert (tmp_path / "one.jpg").exists()
    assert (tmp_path / "three.jpg").exists()
    assert not (tmp_path / "two.jpg").exists()
    assert report["summary"]["downloaded"] == 2
    assert report["summary"]["failed"] == 1
    failures = [d for d in report["details"] if d["status"] == "failed"]
    assert len(failures) == 1 and failures[0]["error"]


def test_failed_asset_is_not_indexed(tmp_path):
    bad = FakeAsset("a2", "two.jpg", dt(2024, 1, 3), raise_times=99)
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", [bad])]))
    dl = make_downloader(api, folder_structure="flat", max_retries=1)

    dl.download(destination=tmp_path)

    data = json.loads((tmp_path / INDEX_FILENAME).read_text())
    assert "a2" not in data["assets"]


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def test_dry_run_downloads_nothing(tmp_path):
    assets = [
        FakeAsset("a1", "one.jpg", dt(2024, 1, 2)),
        FakeAsset("a2", "two.jpg", dt(2024, 1, 3)),
    ]
    api = FakeAPI(FakePhotosService([FakeAlbum("Library", assets)]))
    dl = make_downloader(api, folder_structure="date", dry_run=True)

    report = dl.download(destination=tmp_path / "out")

    assert all(not a.download_calls for a in assets)
    assert not (tmp_path / "out").exists() or list((tmp_path / "out").iterdir()) == []
    assert report["summary"]["dry_run"] is True
    assert report["summary"]["would_download"] == 2
    assert len(report["plan"]) == 2


# ---------------------------------------------------------------------------
# Recently deleted
# ---------------------------------------------------------------------------
def test_recently_deleted_only_included_behind_flag(tmp_path):
    library = FakeAlbum("Library", [FakeAsset("a1", "one.jpg", dt(2024, 1, 2))])
    deleted = FakeAlbum(
        "Recently Deleted", [FakeAsset("a9", "gone.jpg", dt(2024, 1, 5))]
    )
    api = FakeAPI(FakePhotosService([library, deleted], all_album=library))

    without = make_downloader(api, folder_structure="flat")
    assert [a.id for _, a in without.iter_assets()] == ["a1"]

    api2 = FakeAPI(FakePhotosService([library, deleted], all_album=library))
    with_deleted = make_downloader(api2, folder_structure="flat", include_deleted=True)
    assert sorted(a.id for _, a in with_deleted.iter_assets()) == ["a1", "a9"]


# ---------------------------------------------------------------------------
# Auth composition
# ---------------------------------------------------------------------------
def test_authenticate_delegates_to_download_manager(monkeypatch):
    dl = PhotosDownloader(email="user@example.com")
    sentinel = object()
    called = []

    def _fake_auth():
        called.append(True)
        dl._auth_manager.api = sentinel

    monkeypatch.setattr(dl._auth_manager, "authenticate", _fake_auth)

    dl.authenticate()

    assert called == [True]
    assert dl.api is sentinel


def test_email_required(monkeypatch):
    monkeypatch.delenv("ICLOUD_EMAIL", raising=False)
    with pytest.raises(ValueError):
        PhotosDownloader()


# ---------------------------------------------------------------------------
# Streaming plan -> execute, expired URLs
# ---------------------------------------------------------------------------
def test_downloads_start_before_enumeration_finishes(tmp_path):
    assets = [FakeAsset(f"a{i}", f"{i}.jpg", dt(2024, 1, 2)) for i in range(50)]
    dl = make_downloader(
        FakeAPI(FakePhotosService([FakeAlbum("Library", assets)])),
        folder_structure="flat",
        max_workers=2,
    )
    executed_when_yielded = []
    original_execute = dl._execute_plan
    executed = []

    def counting_execute(plan):
        executed.append(plan.asset_id)
        return original_execute(plan)

    def slow_iter_assets(**_kwargs):
        for asset in assets:
            executed_when_yielded.append(len(executed))
            yield "Library", asset

    dl._execute_plan = counting_execute
    dl.iter_assets = slow_iter_assets

    report = dl.download(destination=tmp_path)

    assert executed_when_yielded[-1] > 0
    assert report["summary"]["downloaded"] == 50


def test_expired_url_410_is_retried():
    dl = make_downloader(FakeAPI(None), max_retries=3)
    calls = []

    def expired():
        calls.append(1)
        raise RuntimeError("Apple no longer serves https://x (410)")

    with pytest.raises(Exception):
        dl._with_retry(expired, "asset")

    assert len(calls) == 3


def test_retry_wait_is_capped_for_long_resumed_downloads(monkeypatch):
    dl = make_downloader(FakeAPI(None), max_retries=30)
    waits = []
    monkeypatch.setattr("ifetch.photos.time.sleep", waits.append)
    failures = iter(range(8))

    def flaky():
        if next(failures, None) is not None:
            raise Exception("Connection broken: IncompleteRead(100 bytes read)")
        return "done"

    assert dl._with_retry(flaky, "asset") == "done"
    assert waits[:2] == [2, 4]
    assert len(waits) == 8
    assert max(waits) <= 60


def test_index_saved_when_enumeration_fails(tmp_path):
    assets = [FakeAsset(f"a{i}", f"{i}.jpg", dt(2024, 1, 2)) for i in range(3)]
    dl = make_downloader(
        FakeAPI(FakePhotosService([FakeAlbum("Library", assets)])),
        folder_structure="flat",
        max_workers=2,
    )

    def failing_iter_assets(**_kwargs):
        yield from (("Library", asset) for asset in assets)
        raise RuntimeError("Photos service not available")

    dl.iter_assets = failing_iter_assets

    with pytest.raises(RuntimeError):
        dl.download(destination=tmp_path)

    index = PhotosIndex.load(tmp_path)
    assert all(index.get(asset.id) for asset in assets)


# ---------------------------------------------------------------------------
# Streaming, resume and URL refresh
# ---------------------------------------------------------------------------
class ScriptedSession:
    """Answers GETs from a queue of (status, chunks, error_after_chunks)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        status, chunks, fail = self.script.pop(0)
        if status >= 400:
            raise RuntimeError(f"Apple no longer serves {url} ({status})")
        return ChunkedResponse(status, chunks, fail)


class ChunkedResponse:
    def __init__(self, status, chunks, fail):
        self.status_code = status
        self._chunks = chunks
        self._fail = fail

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        self.chunk_size = chunk_size
        yield from self._chunks
        if self._fail:
            raise ConnectionError("Request failed to iCloud")


def one_asset_downloader(tmp_path, session, asset=None, **kwargs):
    asset = asset or FakeAsset("a1", "big.mov", dt(2024, 1, 2), data=b"x")
    service = FakePhotosService([FakeAlbum("Library", [asset])])
    service.session = session
    return make_downloader(FakeAPI(service), folder_structure="flat", max_retries=3, **kwargs)


def test_download_streams_chunks_to_part_then_renames(tmp_path):
    session = ScriptedSession([(200, [b"aaa", b"bbb", b"ccc"], False)])
    dl = one_asset_downloader(tmp_path, session)

    dl.download(destination=tmp_path)

    assert (tmp_path / "big.mov").read_bytes() == b"aaabbbccc"
    assert not list(tmp_path.glob("*.part"))


def test_every_get_carries_a_timeout(tmp_path):
    session = ScriptedSession([(200, [b"a"], False)])
    dl = one_asset_downloader(tmp_path, session)

    dl.download(destination=tmp_path)

    assert session.calls and all("timeout" in kw for _, kw in session.calls)


def test_retry_resumes_with_range_header(tmp_path):
    session = ScriptedSession([(200, [b"aaaa"], True), (206, [b"bbbb"], False)])
    dl = one_asset_downloader(tmp_path, session)

    report = dl.download(destination=tmp_path)

    assert session.calls[0][1].get("headers") is None
    assert session.calls[1][1]["headers"] == {"Range": "bytes=4-"}
    assert (tmp_path / "big.mov").read_bytes() == b"aaaabbbb"
    assert report["summary"]["failed"] == 0


def test_retry_restarts_when_server_ignores_range(tmp_path):
    session = ScriptedSession([(200, [b"aaaa"], True), (200, [b"aaaabbbb"], False)])
    dl = one_asset_downloader(tmp_path, session)

    dl.download(destination=tmp_path)

    assert (tmp_path / "big.mov").read_bytes() == b"aaaabbbb"


class FakeLibrary:
    def __init__(self, fresh_asset):
        self.zone_id = {"zoneName": "PrimarySync"}
        self.lookups = []
        self._fresh = fresh_asset
        self._client = self

    def lookup(self, record_names, zone_id):
        self.lookups.append((record_names, zone_id))
        return type("Lookup", (), {"records": ["rec"]})()

    def _asset_from_uploaded_records(self, records):
        return self._fresh


def test_retry_uses_url_from_refreshed_records(tmp_path):
    stale = FakeAsset("a1", "big.mov", dt(2024, 1, 2), data=b"x")
    fresh = FakeAsset("a1", "big.mov", dt(2024, 1, 2), data=b"x")
    fresh.versions["original"]["url"] = "https://example.invalid/fresh"
    stale.master_id = "m1"
    stale._library = FakeLibrary(fresh)
    session = ScriptedSession([(410, [], False), (200, [b"ok"], False)])
    dl = one_asset_downloader(tmp_path, session, asset=stale)

    report = dl.download(destination=tmp_path)

    assert [url for url, _ in session.calls] == [
        "https://example.invalid/a1/original",
        "https://example.invalid/fresh",
    ]
    assert stale._library.lookups[0][0] == ["m1", "a1"]
    assert report["summary"]["failed"] == 0


def test_stale_url_is_refreshed_before_first_attempt(tmp_path, monkeypatch):
    stale = FakeAsset("a1", "big.mov", dt(2024, 1, 2), data=b"x")
    fresh = FakeAsset("a1", "big.mov", dt(2024, 1, 2), data=b"x")
    fresh.versions["original"]["url"] = "https://example.invalid/fresh"
    stale.master_id = "m1"
    stale._library = FakeLibrary(fresh)
    session = ScriptedSession([(200, [b"ok"], False)])
    dl = one_asset_downloader(tmp_path, session, asset=stale)
    monkeypatch.setattr("ifetch.photos.URL_MAX_AGE_SECONDS", -1)

    dl.download(destination=tmp_path)

    assert session.calls[0][0] == "https://example.invalid/fresh"
