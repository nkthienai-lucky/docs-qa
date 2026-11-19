import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src" / "docsqa"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_no_framework_imports() -> None:
    imported = [name for path in SRC.rglob("*.py") for name in _imports(path)] if SRC.exists() else []
    assert not any(name == "langchain" or name.startswith("langchain.") for name in imported)
    assert not any(name == "llama_index" or name.startswith("llama_index.") for name in imported)


def test_web_app_does_not_import_judge() -> None:
    for filename in ("api.py", "ui.py"):
        path = SRC / filename
        if not path.exists():
            pytest.skip(f"{filename} is introduced by a later task")
        imported = _imports(path)
        assert "docsqa.eval" not in imported
        assert not any(name.startswith("docsqa.eval.") for name in imported)
