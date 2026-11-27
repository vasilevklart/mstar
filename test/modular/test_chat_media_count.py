"""A chat's attachments count against MSTAR_MAX_MEDIA_FILES before any is saved or fetched."""

import pytest

from mstar.api_server import media_io
from mstar.api_server.openai.adapters import flatten_messages


def _image(url):
    return {"type": "image_url", "image_url": {"url": url}}


def test_too_many_remote_urls_are_refused_before_any_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv("MSTAR_MAX_MEDIA_FILES", "2")
    fetched = []

    def fetch(url, upload_dir, allow_remote=False):
        fetched.append(url)
        return "image", str(upload_dir / f"{len(fetched)}.png")

    monkeypatch.setattr(media_io, "resolve_media_ref", fetch)
    messages = [{"role": "user", "content": [_image(f"https://example.com/{i}.png") for i in range(3)]}]
    with pytest.raises(ValueError, match=r"attaches 3 media files; the limit is 2 \(MSTAR_MAX_MEDIA_FILES\)"):
        flatten_messages(messages, tmp_path, allow_remote=True)
    assert fetched == [], "URLs were fetched before the request was refused"


def test_too_many_inline_files_leave_nothing_on_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("MSTAR_MAX_MEDIA_FILES", "2")
    audio = {"type": "input_audio", "input_audio": {"data": "aGk=", "format": "wav"}}
    images = [_image("data:image/png;base64,aGk="), _image("data:image/png;base64,aGk=")]
    messages = [{"role": "user", "content": images}, {"role": "user", "content": [audio]}]
    with pytest.raises(ValueError, match="attaches 3 media files; the limit is 2"):
        flatten_messages(messages, tmp_path)
    assert list(tmp_path.iterdir()) == []

    # at the cap, and an empty URL is not an attachment
    messages = [{"role": "user", "content": [*images, _image(""), {"type": "text", "text": "hi"}]}]
    _, file_paths, _, _ = flatten_messages(messages, tmp_path)
    assert sum(len(paths) for paths in file_paths.values()) == 2
