"""Bounded, credential-safe HTTP boundary for runtime API providers."""

from __future__ import annotations

import json
import math
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class ProviderHTTPError(ValueError):
    """Provider transport or payload failure without upstream details."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def provider_endpoint(host: str, endpoint: str) -> str:
    result = None
    try:
        parsed = urlsplit(host)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or "?" in host
            or "#" in host
            or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in host)
            or (parsed.port is not None and not 0 < parsed.port <= 65535)
            or endpoint not in {"chat/completions", "embeddings", "rerank"}
        ):
            raise ValueError
        path = parsed.path.rstrip("/")
        for suffix in ("/chat/completions", "/embeddings", "/rerank"):
            if path.endswith(suffix):
                path = path[: -len(suffix)]
                break
        path = (path or "/v1") + "/" + endpoint
        result = urlunsplit((parsed.scheme, parsed.netloc.lower(), path, "", ""))
    except (ValueError, TypeError):
        pass
    if result is None:
        raise ProviderHTTPError("invalid provider endpoint")
    return result


def open_provider_request(request: Request, timeout: float):
    return build_opener(NoRedirect()).open(request, timeout=timeout)


def _reject_constant(value):
    raise ValueError("nonfinite JSON")


def post_json(url: str, token: str, payload: dict, timeout: float) -> dict:
    result = None
    try:
        # Validate even direct transport callers, before constructing headers or opening a socket.
        provider_endpoint(url, "embeddings")
        if (
            not token
            or token != token.strip()
            or any(ord(c) < 32 or ord(c) == 127 for c in token)
        ):
            raise ValueError
        if not math.isfinite(timeout) or not 0 < timeout <= 120:
            raise ValueError
        request = Request(
            url,
            data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        with open_provider_request(request, timeout) as response:
            if not 200 <= response.getcode() < 300:
                raise ValueError
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError
        body = json.loads(raw, parse_constant=_reject_constant)
        if not isinstance(body, dict):
            raise TypeError
        result = body
    except Exception:  # noqa: BLE001, S110 — upstream errors may contain secrets
        # Raise outside the handler: even hidden exception context can retain secrets.
        pass
    if result is None:
        raise ProviderHTTPError("provider request failed")
    return result
