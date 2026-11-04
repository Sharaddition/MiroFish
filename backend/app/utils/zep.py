"""Shared Zep Cloud client, request limits, and retry policy."""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any, Callable, TypeVar

import httpx
from zep_cloud.client import Zep
from zep_cloud.core.api_error import ApiError as ZepApiError

from ..config import Config
from .logger import get_logger

logger = get_logger("mirofish.zep")

T = TypeVar("T")

ZEP_CLOUD_BASE_URL = "https://api.getzep.com/api/v2"
# Keep request behavior aligned with the zep-cloud 3.25.0 SDK default that
# MiroFish used before introducing the shared client. This is an internal
# integration policy, not a deployment setting users need to tune.
ZEP_HTTP_REQUEST_TIMEOUT_SECONDS = 60.0
# Zep ingestion is asynchronous and may take several minutes. Preserve the
# original GraphBuilder deadline while keeping it separate from HTTP timeout.
# A big simulation can queue hundreds of episodes that Zep works through at its own pace (about one a minute was
# seen), so the wait at the end of a run can be raised with ZEP_INGESTION_WAIT_TIMEOUT_SECONDS in .env.
def _wait_timeout_seconds(default: int = 600) -> int:
    try:
        value = int(float(os.environ.get("ZEP_INGESTION_WAIT_TIMEOUT_SECONDS") or 0))
    except ValueError:
        value = 0
    return value if value > 0 else default


ZEP_INGESTION_WAIT_TIMEOUT_SECONDS = _wait_timeout_seconds()
MAX_ZEP_SEARCH_QUERY_CHARS = 400
MAX_ZEP_SEARCH_RESULTS = 50


def normalize_zep_search_query(query: Any) -> str:
    """Return a non-empty query within Zep Cloud's endpoint limit."""

    if not isinstance(query, str):
        raise ValueError("Zep search query must be a string")
    normalized = query.strip()
    if not normalized:
        raise ValueError("Zep search query must not be empty")
    return normalized[:MAX_ZEP_SEARCH_QUERY_CHARS]


def normalize_zep_search_limit(limit: Any) -> int:
    """Clamp a search result limit to the current Zep Cloud contract."""

    try:
        normalized = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("Zep search limit must be an integer") from exc
    if normalized < 1:
        raise ValueError("Zep search limit must be at least 1")
    return min(normalized, MAX_ZEP_SEARCH_RESULTS)


@lru_cache(maxsize=4)
def _cached_zep_client(api_key: str, timeout: float) -> Zep:
    return Zep(
        api_key=api_key,
        base_url=ZEP_CLOUD_BASE_URL,
        timeout=timeout,
    )


def get_zep_client(api_key: str | None = None, timeout: float | None = None) -> Zep:
    """Return a process-shared, explicitly configured Zep Cloud client."""

    if Config.GRAPH_BACKEND == "graphiti":
        # Self-hosted Graphiti + FalkorDB behind the same SDK calls (no Zep account or key involved)
        from .graphiti_backend import get_graphiti_client

        return get_graphiti_client()

    # zep-cloud gives ZEP_API_URL precedence even when base_url is explicit.
    # Reject it so this Cloud-only integration cannot silently target a
    # self-hosted or compatibility endpoint.
    if os.environ.get("ZEP_API_URL"):
        raise ValueError("ZEP_API_URL is unsupported; unset it to use Zep Cloud")

    normalized_key = (api_key or Config.ZEP_API_KEY or "").strip()
    if not normalized_key:
        raise ValueError("ZEP_API_KEY 未配置")

    request_timeout = float(
        timeout if timeout is not None else ZEP_HTTP_REQUEST_TIMEOUT_SECONDS
    )
    if request_timeout <= 0:
        raise ValueError("Zep request timeout must be greater than 0")
    return _cached_zep_client(normalized_key, request_timeout)


def clear_zep_client_cache() -> None:
    """Clear cached clients. Intended for tests and controlled reconfiguration."""

    _cached_zep_client.cache_clear()
    if Config.GRAPH_BACKEND == "graphiti":
        from .graphiti_backend import reset_graphiti_client

        reset_graphiti_client()


def is_retryable_zep_error(error: BaseException) -> bool:
    """Return whether a failed *read* is safe and useful to retry."""

    if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True
    if isinstance(error, ZepApiError):
        status_code = error.status_code
        return status_code in {408, 429} or (
            status_code is not None and 500 <= status_code <= 599
        )
    return False


def is_pre_rejection_write_error(error: BaseException) -> bool:
    """Return whether a failed *write* was rejected before Zep processed it.

    A 429 is a rate-limiter rejection and a connect failure means the request
    never reached the API, so replaying the write cannot duplicate facts.
    Timeouts and 5xx responses are ambiguous outcomes and must never be
    replayed, because graph writes have no idempotency key.
    """

    if isinstance(error, httpx.ConnectError):
        return True
    if isinstance(error, ZepApiError):
        return error.status_code == 429
    return False


def _retry_after_seconds(error: BaseException) -> float | None:
    if not isinstance(error, ZepApiError) or not error.headers:
        return None
    value = next(
        (
            header_value
            for header_name, header_value in error.headers.items()
            if header_name.lower() == "retry-after"
        ),
        None,
    )
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _header(error: BaseException, name: str) -> str | None:
    headers = getattr(error, "headers", None)
    if not headers:
        return None
    return next((str(v) for k, v in headers.items() if str(k).lower() == name), None)


def rate_limit_reset_seconds(error: BaseException, now: Callable[[], float] | None = None) -> float | None:
    """Seconds until Zep's rate-limit window resets, from ``x-ratelimit-reset`` (an epoch time), for a 429.

    The FREE plan answers 429 with ``retry-after: 0`` and ``x-ratelimit-remaining: 0``: retrying at once is refused
    again, so the reset time is the only honest wait. ``None`` when the error carries no usable reset header.
    """

    if not isinstance(error, ZepApiError) or error.status_code != 429:
        return None
    try:
        reset = float(_header(error, "x-ratelimit-reset") or "")
    except ValueError:
        return None
    # one extra second: the window opens at the reset second, and clocks differ a little
    return max(0.0, reset - (now or time.time)()) + 1.0


def retry_delay_seconds(
    error: BaseException,
    attempt: int,
    initial_delay: float,
    max_delay: float,
    now: Callable[[], float] | None = None,
) -> float:
    """How long to wait before retrying ``error``: Zep's ``retry-after`` when it is positive, else the rate-limit
    window's reset, else exponential backoff. Capped at ``max_delay``."""

    retry_after = _retry_after_seconds(error)
    if retry_after is not None and retry_after > 0:
        wait = retry_after
    else:
        wait = rate_limit_reset_seconds(error, now)
        if wait is None:
            wait = initial_delay * (2 ** (attempt - 1))
    return min(wait, max_delay)


def _describe(error: BaseException) -> str:
    status = getattr(error, "status_code", None)
    return f"{type(error).__name__} {status}" if status else type(error).__name__


def call_zep_read_with_retry(
    operation: Callable[[], T],
    *,
    operation_name: str,
    max_attempts: int = 3,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry a safe Zep read only for transport, 408, 429, or 5xx errors."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == max_attempts or not is_retryable_zep_error(error):
                raise

            delay = retry_delay_seconds(error, attempt, initial_delay, max_delay)
            logger.warning(
                "Zep %s attempt %s/%s failed (%s); retrying in %.1fs",
                operation_name,
                attempt,
                max_attempts,
                _describe(error),
                delay,
            )
            sleep(delay)

    raise AssertionError("unreachable")


def call_zep_write_with_pre_rejection_retry(
    operation: Callable[[], T],
    *,
    operation_name: str,
    max_attempts: int = 4,
    initial_delay: float = 2.0,
    max_delay: float = 30.0,
    deadline: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Replay a Zep write only when it was rejected before processing.

    Safe to replay: 429 rejections and connect failures (nothing was written).
    Never replayed: ambiguous outcomes such as timeouts or 5xx, which keep
    the fail-closed contract for non-idempotent graph writes.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == max_attempts or not is_pre_rejection_write_error(error):
                raise
            if deadline is not None and time.time() >= deadline:
                raise

            delay = retry_delay_seconds(error, attempt, initial_delay, max_delay)
            if deadline is not None:
                delay = min(delay, max(0.0, deadline - time.time()))
            logger.warning(
                "Zep %s attempt %s/%s rejected before processing (%s); "
                "replaying in %.1fs",
                operation_name,
                attempt,
                max_attempts,
                _describe(error),
                delay,
            )
            sleep(delay)

    raise AssertionError("unreachable")
