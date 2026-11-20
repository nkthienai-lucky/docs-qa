import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from docsqa.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


def settings(**kwargs):
    return Settings(_env_file=None, **kwargs)


def test_bind_local_and_cloud():
    from docsqa.serve import resolve_bind

    assert resolve_bind(settings(), {}) == ("127.0.0.1", 8000)
    assert resolve_bind(settings(), {"HOST": "localhost", "PORT": "9000"}) == (
        "localhost",
        9000,
    )
    assert resolve_bind(
        settings(runtime_profile="cloud", runtime_deployment_approved=True),
        {"PORT": "10000"},
    ) == ("0.0.0.0", 10000)


@pytest.mark.parametrize(
    "environ",
    [
        {},
        {"PORT": "0"},
        {"PORT": "65536"},
        {"PORT": "true"},
        {"PORT": "3.0"},
        {"PORT": " 8"},
    ],
)
def test_cloud_rejects_bad_port(environ):
    from docsqa.runtime import RuntimeConfigurationError
    from docsqa.serve import resolve_bind

    with pytest.raises(RuntimeConfigurationError):
        resolve_bind(
            settings(runtime_profile="cloud", runtime_deployment_approved=True), environ
        )


def test_public_bind_interlock():
    from docsqa.runtime import RuntimeConfigurationError
    from docsqa.serve import resolve_bind

    with pytest.raises(RuntimeConfigurationError):
        resolve_bind(settings(runtime_profile="cloud"), {"PORT": "10000"})
    with pytest.raises(RuntimeConfigurationError):
        resolve_bind(settings(), {"HOST": "0.0.0.0"})


def test_upload_storage_checks_and_cleanup(tmp_path):
    from docsqa.serve import validate_upload_dir

    local = tmp_path / "local"
    assert validate_upload_dir(settings(runtime_upload_dir=str(local))) == local
    assert list(local.iterdir()) == []
    assert (
        validate_upload_dir(
            settings(runtime_profile="cloud", runtime_upload_dir=str(local))
        )
        == local
    )
    assert list(local.iterdir()) == []


def test_cloud_storage_missing_relative_symlink_unwritable(tmp_path, monkeypatch):
    from docsqa.runtime import RuntimeConfigurationError
    from docsqa.serve import validate_upload_dir

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    for path in ["relative", str(tmp_path / "missing"), str(link), str(link / "child")]:
        with pytest.raises(RuntimeConfigurationError):
            validate_upload_dir(
                settings(runtime_profile="cloud", runtime_upload_dir=path)
            )
    monkeypatch.setattr(
        "docsqa.serve.tempfile.NamedTemporaryFile",
        lambda **kwargs: (_ for _ in ()).throw(PermissionError("sentinel-path")),
    )
    with pytest.raises(RuntimeConfigurationError) as caught:
        validate_upload_dir(
            settings(runtime_profile="cloud", runtime_upload_dir=str(real))
        )
    assert caught.value.__context__ is None
    assert list(real.iterdir()) == []


def test_cloud_validation_precedes_composition(monkeypatch):
    from docsqa.serve import create_runtime_app

    monkeypatch.setattr(
        "docsqa.serve.get_settings", lambda: settings(runtime_profile="cloud")
    )
    monkeypatch.setattr(
        "docsqa.serve.build_runtime_app",
        lambda s: pytest.fail("composition must not run"),
    )
    with pytest.raises(ValueError):
        create_runtime_app()


def test_packaging_contract():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert any(d.startswith("uvicorn") for d in project["dependencies"])
    assert project["scripts"]["docsqa-api"] == "docsqa.serve:main"
    docker = (ROOT / "Dockerfile").read_text()
    assert "uv sync --frozen" in docker and "scripts/container_start.sh" in docker
    assert "ARG " not in docker and "ollama pull" not in docker
    ignored = (ROOT / ".dockerignore").read_text()
    for pattern in [".env*", ".git", ".venv", "*.pdf", "data", "eval"]:
        assert pattern in ignored
    render = (ROOT / "render.yaml").read_text()
    for text in [
        "/var/data",
        "/var/data/uploads",
        "RUNTIME_DEPLOYMENT_APPROVED",
        '"false"',
        "numInstances: 1",
    ]:
        assert text in render
    startup = (ROOT / "scripts/container_start.sh").read_text()
    assert "mountpoint -q /var/data" in startup
    assert "docsqa-api" in startup


def test_imports_offline():
    result = subprocess.run(
        [sys.executable, "-c", "import docsqa.api, docsqa.ui, docsqa.serve"],
        capture_output=True,
        timeout=60,
        check=False,
        env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
    )
    assert result.returncode == 0


def test_startup_script_cloud_without_mount_refuses_before_api():
    result = subprocess.run(
        ["sh", str(ROOT / "scripts/container_start.sh")],
        env={
            "PATH": "/usr/bin:/bin",
            "RUNTIME_PROFILE": "cloud",
            "RUNTIME_UPLOAD_DIR": "/var/data/uploads",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0
    assert result.stderr.strip() == "persistent upload mount required"
