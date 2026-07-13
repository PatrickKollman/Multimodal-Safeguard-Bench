"""
Download resilience: retry with exponential backoff + automatic transfer-backend fallback.

Env controls
------------
MSBENCH_FAST_TRANSFER=0
    Set before starting a run to disable hf_transfer and xet backends for the
    entire process. Downloads fall back to plain HTTPS — slower but immune to
    stream-drop errors from those backends.

    Example:
        MSBENCH_FAST_TRANSFER=0 python -m msbench.run --config configs/mvp.yaml --smoke

    The retry wrapper also sets this automatically on its final attempt after
    repeated transfer-backend failures, so partial runs recover without manual
    intervention.
"""
from __future__ import annotations
import logging
import os
import time

log = logging.getLogger(__name__)

_MAX_ATTEMPTS = 4   # 3 retries with fast backends, then 1 with plain HTTPS
_BASE_DELAY = 5.0   # seconds; doubles each retry


def _is_transient_download_error(exc: BaseException) -> bool:
    """True for stream/network failures from hf_transfer or xet that are worth retrying."""
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return any(phrase in msg for phrase in (
            "receiver dropped",
            "failed to send data",
            "internal writer error",
            "xet",
        ))
    # requests / urllib3 connection errors
    try:
        import requests.exceptions as _re
        if isinstance(exc, (_re.ConnectionError, _re.Timeout, _re.ChunkedEncodingError)):
            return True
    except ImportError:
        pass
    try:
        import urllib3.exceptions as _u3
        if isinstance(exc, (_u3.ProtocolError, _u3.NewConnectionError)):
            return True
    except ImportError:
        pass
    return False


def _disable_fast_transfer() -> None:
    """Switch off hf_transfer and xet for the remainder of this process."""
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    log.warning(
        "Fast transfer backends disabled (HF_HUB_ENABLE_HF_TRANSFER=0, HF_HUB_DISABLE_XET=1). "
        "Subsequent downloads use plain HTTPS. "
        "To opt out from the start, set MSBENCH_FAST_TRANSFER=0 before running."
    )


def load_with_retry(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) — typically a from_pretrained() call — and retry on
    transient transfer-backend failures with exponential backoff.

    Attempts 1–(_MAX_ATTEMPTS-1): retry with fast backends enabled.
    Final attempt: disables fast backends first, then retries with plain HTTPS.
    Non-transient errors (access denied, config parse failures, …) are re-raised
    immediately without retrying.
    """
    if os.environ.get("MSBENCH_FAST_TRANSFER", "").strip() == "0":
        _disable_fast_transfer()

    delay = _BASE_DELAY
    last_exc: BaseException | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        if attempt == _MAX_ATTEMPTS and last_exc is not None:
            log.warning("Final retry: disabling fast transfer backends before last attempt.")
            _disable_fast_transfer()
        try:
            return fn(*args, **kwargs)
        except BaseException as exc:
            if not _is_transient_download_error(exc):
                raise
            last_exc = exc
            if attempt < _MAX_ATTEMPTS:
                log.warning(
                    f"Transient download error (attempt {attempt}/{_MAX_ATTEMPTS}), "
                    f"retrying in {delay:.0f}s:\n  {exc}"
                )
                time.sleep(delay)
                delay *= 2

    raise last_exc  # type: ignore[misc]
