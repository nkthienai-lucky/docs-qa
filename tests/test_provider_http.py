import io
import json
import traceback
from urllib.error import HTTPError

import pytest


@pytest.mark.parametrize(
    "host,want",
    [
        ("https://example.invalid", "https://example.invalid/v1/embeddings"),
        ("https://example.invalid/v1/", "https://example.invalid/v1/embeddings"),
        (
            "https://example.invalid/gateway/v1/chat/completions/",
            "https://example.invalid/gateway/v1/embeddings",
        ),
        ("https://example.invalid/embeddings", "https://example.invalid/v1/embeddings"),
    ],
)
def test_endpoint_normalization(host, want):
    from docsqa.provider_http import provider_endpoint

    assert provider_endpoint(host, "embeddings") == want


@pytest.mark.parametrize(
    "host",
    [
        "https://u:p@host",
        "https://host?x=1",
        "https://host#x",
        "https:///bad",
        "file:///tmp/foo",
        "https://host/has space",
        "https://host:bad",
    ],
)
def test_endpoint_rejects_unsafe_values(host):
    from docsqa.provider_http import ProviderHTTPError, provider_endpoint

    with pytest.raises(ProviderHTTPError):
        provider_endpoint(host, "embeddings")


def install_response(monkeypatch, body, status=200):
    import docsqa.provider_http as http

    captured = {}

    class Response(io.BytesIO):
        def getcode(self):
            return status

    def open_request(request, timeout):
        captured.update(
            url=request.full_url,
            headers=dict(request.header_items()),
            payload=json.loads(request.data),
            timeout=timeout,
        )
        return Response(body)

    monkeypatch.setattr(http, "open_provider_request", open_request)
    return captured


def test_post_json_request(monkeypatch):
    from docsqa.provider_http import post_json

    captured = install_response(monkeypatch, b'{"ok": true}')
    assert post_json(
        "https://host/v1/embeddings", "private-token", {"input": ["a"]}, 3
    ) == {"ok": True}
    assert captured["headers"]["Authorization"] == "Bearer private-token"
    assert captured["payload"] == {"input": ["a"]}
    assert captured["timeout"] == 3


@pytest.mark.parametrize(
    "body,status",
    [
        (b"[]", 200),
        (b"bad-json", 200),
        (b'{"x":NaN}', 200),
        (b"{}", 302),
        (b"{}", 500),
        (b"x" * (8 * 1024 * 1024 + 1), 200),
    ],
    ids=["array", "malformed", "nan", "redirect", "server-error", "oversize"],
)
def test_post_json_rejects_invalid_response(monkeypatch, body, status):
    from docsqa.provider_http import ProviderHTTPError, post_json

    install_response(monkeypatch, body, status)
    with pytest.raises(ProviderHTTPError):
        post_json("https://host/v1/embeddings", "token", {}, 3)


@pytest.mark.parametrize("code", [301, 302, 307, 308])
def test_redirect_handler_never_follows(code):
    from docsqa.provider_http import NoRedirect

    assert (
        NoRedirect().redirect_request(None, None, code, "redirect", {}, "https://other")
        is None
    )


def test_transport_exception_has_no_raw_chain(monkeypatch):
    import docsqa.provider_http as http

    def fail(*args):
        raise HTTPError("https://sentinel-url", 500, "sentinel-body", {}, None)

    monkeypatch.setattr(http, "open_provider_request", fail)
    with pytest.raises(http.ProviderHTTPError) as caught:
        http.post_json("https://host/v1/embeddings", "sentinel-token", {}, 3)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    rendered = "".join(traceback.format_exception(caught.value))
    assert "sentinel-url" not in rendered
    assert "sentinel-body" not in rendered


def test_transport_reads_only_cap_plus_one(monkeypatch):
    import docsqa.provider_http as http

    reads = []

    class Response(io.BytesIO):
        def getcode(self):
            return 200

        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    monkeypatch.setattr(http, "open_provider_request", lambda *args: Response(b"{}"))
    assert http.post_json("https://host/v1/embeddings", "token", {}, 3) == {}
    assert reads == [http.MAX_RESPONSE_BYTES + 1]


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_opener_denies_redirect_before_second_send(monkeypatch, code):
    from email.message import Message
    from urllib.request import BaseHandler, Request, build_opener
    from urllib.response import addinfourl

    from docsqa.provider_http import NoRedirect

    sends = []

    class FakeHTTPS(BaseHandler):
        handler_order = 100

        def https_open(self, request):
            sends.append(request.full_url)
            headers = Message()
            headers["Location"] = "https://other.invalid/steal"
            response = addinfourl(
                io.BytesIO(b"sentinel-body"), headers, request.full_url, code
            )
            response.msg = "redirect"
            return response

    opener = build_opener(FakeHTTPS(), NoRedirect())
    with pytest.raises(HTTPError):
        opener.open(
            Request(
                "https://host.invalid/v1",
                headers={"Authorization": "Bearer sentinel-token"},
            )
        )
    assert sends == ["https://host.invalid/v1"]


@pytest.mark.parametrize(
    "error", [TimeoutError("sentinel-body"), ValueError("sentinel-token")]
)
def test_transport_timeout_and_parse_diagnostics_hidden(monkeypatch, capsys, error):
    import docsqa.provider_http as http

    def fail(*args):
        raise error

    monkeypatch.setattr(http, "open_provider_request", fail)
    request_args = (
        "https://sentinel-url",
        "sentinel-token",
        {"input": "sentinel-input"},
        3,
    )
    with pytest.raises(http.ProviderHTTPError) as caught:
        http.post_json(*request_args)
    rendered = "".join(traceback.format_exception(caught.value))
    output = capsys.readouterr()
    assert "sentinel" not in rendered + output.out + output.err
    assert caught.value.__context__ is caught.value.__cause__ is None


@pytest.mark.parametrize(
    "url,token",
    [
        ("https://user:password@host/v1", "token"),
        ("https://host/v1?private=yes", "token"),
        ("https://host/v1", "token\r\nInjected: header"),
        ("https://host/v1", ""),
    ],
)
def test_transport_rejects_unsafe_credentials_before_open(monkeypatch, url, token):
    import docsqa.provider_http as http

    calls = []
    monkeypatch.setattr(http, "open_provider_request", lambda *args: calls.append(args))
    with pytest.raises(http.ProviderHTTPError):
        http.post_json(url, token, {}, 3)
    assert calls == []
