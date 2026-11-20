import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/runtime_api_smoke.py"


def smoke():
    spec = importlib.util.spec_from_file_location("runtime_api_smoke", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_without_flag_does_not_construct(monkeypatch, capsys):
    module = smoke()
    monkeypatch.setattr(
        module,
        "Settings",
        lambda **kwargs: pytest.fail("settings construction forbidden"),
    )
    assert module.main([]) == 0
    assert (
        capsys.readouterr().out == "Runtime API smoke skipped (network not enabled).\n"
    )


def test_invalid_cloud_no_components(monkeypatch, capsys):
    module = smoke()
    monkeypatch.setattr(
        module,
        "build_runtime_components",
        lambda s: pytest.fail("invalid settings must not construct"),
    )
    monkeypatch.setattr(
        module, "Settings", lambda **kwargs: SimpleNamespace(runtime_profile="local")
    )
    assert module.main(["--allow-network"]) == 1
    assert "failed" in capsys.readouterr().out


@pytest.mark.parametrize("bad", [False, True])
def test_mocked_synthetic_semantics(monkeypatch, capsys, bad):
    module = smoke()
    settings = SimpleNamespace(
        runtime_profile="cloud", runtime_deployment_approved=True
    )
    received = []

    def load(**kwargs):
        received.append(kwargs)
        return settings

    monkeypatch.setattr(module, "Settings", load)
    monkeypatch.setattr(module, "validate_runtime_profile", lambda s: None)
    monkeypatch.setattr(
        module,
        "build_runtime_components",
        lambda s: SimpleNamespace(
            generator=SimpleNamespace(
                generate=lambda messages: (
                    "wrong-private-answer" if bad else "SYNTHETIC_OK"
                )
            ),
            embedder=SimpleNamespace(
                embed_documents=lambda texts: [[1.0, 0.0] for _ in texts]
            ),
            reranker=SimpleNamespace(
                rerank=lambda query, hits, top_k: list(reversed(hits))
            ),
        ),
    )
    assert module.main(["--allow-network"]) == (1 if bad else 0)
    assert received == [{"_env_file": None}]
    out = capsys.readouterr().out
    assert ("2 passed, 1 failed" if bad else "3 passed, 0 failed") in out
    assert "private" not in out and "SYNTHETIC_OK" not in out


def test_smoke_transport_error_never_printed(monkeypatch, capsys):
    module = smoke()

    def fail(**kwargs):
        raise RuntimeError("sentinel-token https://private corpus-text")

    monkeypatch.setattr(module, "Settings", fail)
    assert module.main(["--allow-network"]) == 1
    output = capsys.readouterr()
    assert "sentinel" not in output.out + output.err
    assert "https://" not in output.out + output.err


def test_network_flag_requires_deployment_approval(monkeypatch, capsys):
    module = smoke()
    monkeypatch.setattr(
        module,
        "Settings",
        lambda **kwargs: SimpleNamespace(
            runtime_profile="cloud", runtime_deployment_approved=False
        ),
    )
    monkeypatch.setattr(module, "validate_runtime_profile", lambda s: None)
    monkeypatch.setattr(
        module,
        "build_runtime_components",
        lambda s: pytest.fail("components must not be built without approval"),
    )
    assert module.main(["--allow-network"]) == 1
    assert "failed" in capsys.readouterr().out
