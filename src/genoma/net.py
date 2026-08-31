"""Shared HTTP helper: retry-with-backoff on transient errors, still fail-fast.

External REST calls (Ensembl VEP, Phen2Gene, PubMed, OpenFDA, ClinVar) hit
rate-limit (429) and server-side (5xx) blips routinely. This helper retries
ONLY those transient classes with exponential backoff: every other error
propagates on first occurrence, and exhausted retries re-raise the last error.
No silent fallback, ever.
"""

from __future__ import annotations

import http.client
import ssl
import time
import urllib.error
import urllib.request

# Rate-limit + server-side transients worth retrying.
TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})


# NixOS quirk: the system Python here loads ZERO CA certs by default, so
# every HTTPS urlopen() fails with CERTIFICATE_VERIFY_FAILED. certifi ships a
# current CA bundle in the venv — prefer it (fall back to the OS default
# context when certifi is absent). One shared context for all callers.
def _default_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError, FileNotFoundError, ssl.SSLError:
        return ssl.create_default_context()


_SSL_CONTEXT = _default_context()


def urlopen_with_retry(
    url_or_req: str | urllib.request.Request,
    *,
    timeout: float = 20,
    attempts: int = 4,
    backoff_s: float = 2.0,
) -> http.client.HTTPResponse:
    """urlopen() with exponential backoff on transient HTTP/network errors.

    Retried: HTTP 429/500/502/503/504, URLError (connection/DNS blips),
    TimeoutError. Not retried: 4xx client errors and anything else — those
    raise immediately. When attempts are exhausted the final error raises, so
    callers keep their fail-fast contract.

    Defaults (4 attempts, 2.0s base ×1.5 → ~2/3/4.5s) survive a VEP/Entrez
    rate-limit burst from the parallel scorer without hammering the endpoint.
    """
    last_err: Exception | None = None
    delay = backoff_s
    for attempt in range(1, attempts + 1):
        try:
            return urllib.request.urlopen(url_or_req, timeout=timeout, context=_SSL_CONTEXT)
        except urllib.error.HTTPError as e:
            if e.code not in TRANSIENT_STATUS:
                raise
            last_err = e
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
        if attempt < attempts:
            time.sleep(delay)
            delay *= 1.5
    # Loop only exits without returning when at least one except ran.
    assert last_err is not None
    raise last_err
