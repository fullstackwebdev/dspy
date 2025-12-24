import base64
import io
import mimetypes
import os
import warnings
from functools import lru_cache
from typing import Any, Union
from urllib.parse import urlparse

import pydantic
import requests

from dspy.adapters.types.base_type import Type


class Video(Type):
    url: str

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    def __init__(self, url: Any = None, *, download: bool = False, **data):
        """Create a Video.

        Parameters
        ----------
        url:
            The video source. Supported values include

            - ``str``: HTTP(S)/GS URL or local file path
            - ``bytes``: raw video bytes
            - ``dict`` with a single ``{"url": value}`` entry (legacy form)
            - already encoded data URI

        download:
            Whether remote URLs should be downloaded to infer their MIME type.

        Any additional keyword arguments are passed to :class:`pydantic.BaseModel`.
        """

        if url is not None and "url" not in data:
            # Support a positional argument while allowing ``url=`` in **data.
            if isinstance(url, dict) and set(url.keys()) == {"url"}:
                # Legacy dict form from previous model validator.
                data["url"] = url["url"]
            else:
                # ``url`` may be a string, bytes.
                data["url"] = url

        if "url" in data:
            # Normalize any accepted input into a base64 data URI or plain URL.
            data["url"] = encode_video(data["url"], download_videos=download)

        # Delegate the rest of initialization to pydantic's BaseModel.
        super().__init__(**data)

    @lru_cache(maxsize=32)
    def format(self) -> list[dict[str, Any]] | str:
        try:
            video_url = encode_video(self.url)
        except Exception as e:
            raise ValueError(f"Failed to format video for DSPy: {e}")
        return [{"type": "video_url", "video_url": {"url": video_url}}]

    @classmethod
    def from_url(cls, url: str, download: bool = False):
        warnings.warn(
            "Video.from_url is deprecated; use Video(url) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(url, download=download)

    @classmethod
    def from_file(cls, file_path: str):
        warnings.warn(
            "Video.from_file is deprecated; use Video(file_path) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(file_path)

    def __str__(self):
        return self.serialize_model()

    def __repr__(self):
        if "base64" in self.url:
            len_base64 = len(self.url.split("base64,")[1])
            video_type = self.url.split(";")[0].split("/")[-1]
            return f"Video(url=data:video/{video_type};base64,<VIDEO_BASE_64_ENCODED({len_base64!s})>)"
        return f"Video(url='{self.url}')"


def is_url(string: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(string)
        return all([result.scheme in ("http", "https", "gs"), result.netloc])
    except ValueError:
        return False


def encode_video(video: Union[str, bytes, dict], download_videos: bool = False) -> str:
    """
    Encode a video or file to a base64 data URI.

    Args:
        video: The video or file to encode. Can be file path, URL, or data URI.
        download_videos: Whether to download videos from URLs.

    Returns:
        str: The data URI of the file or the URL if download_videos is False.

    Raises:
        ValueError: If the file type is not supported.
    """
    if isinstance(video, dict) and "url" in video:
        # NOTE: Not doing other validation for now
        return video["url"]
    elif isinstance(video, str):
        if video.startswith("data:"):
            # Already a data URI
            return video
        elif os.path.isfile(video):
            # File path
            return _encode_video_from_file(video)
        elif is_url(video):
            # URL
            if download_videos:
                return _encode_video_from_url(video)
            else:
                # Return the URL as is
                return video
        else:
            # Unsupported string format
            raise ValueError(f"Unrecognized file string: {video}; If this file type should be supported, please open an issue.")
    elif isinstance(video, bytes):
        # Raw bytes
        return _encode_bytes_video(video)
    elif isinstance(video, Video):
        return video.url
    else:
        print(f"Unsupported video type: {type(video)}")
        raise ValueError(f"Unsupported video type: {type(video)}")


def _encode_video_from_file(file_path: str) -> str:
    """Encode a file from a file path to a base64 data URI."""
    with open(file_path, "rb") as file:
        file_data = file.read()

    # Use mimetypes to guess directly from the file path
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for file: {file_path}")

    encoded_data = base64.b64encode(file_data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_data}"


def _encode_video_from_url(video_url: str) -> str:
    """Encode a file from a URL to a base64 data URI."""
    response = requests.get(video_url)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")

    # Use the content type from the response headers if available
    if content_type:
        mime_type = content_type
    else:
        # Try to guess MIME type from URL
        mime_type, _ = mimetypes.guess_type(video_url)
        if mime_type is None:
            raise ValueError(f"Could not determine MIME type for URL: {video_url}")

    encoded_data = base64.b64encode(response.content).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_data}"


def _encode_bytes_video(video: bytes) -> str:
    """Encode raw video bytes to a base64 data URI."""
    # Guess the MIME type based on the file content
    mime_type = mimetypes.guess_type(f"file.mp4")[0] or "video/mp4"
    encoded_data = base64.b64encode(video).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_data}"


def is_video(obj) -> bool:
    """Check if the object is a video or a valid media file reference."""
    if isinstance(obj, str):
        if obj.startswith("data:"):
            return obj.startswith("data:video/")
        elif os.path.isfile(obj):
            mime_type, _ = mimetypes.guess_type(obj)
            return mime_type and mime_type.startswith("video/")
        elif is_url(obj):
            return True  # For URLs, we assume it could be a video
    return False