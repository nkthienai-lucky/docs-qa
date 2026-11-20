from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from docsqa.settings import Settings
from docsqa.store import EmbedderMeta, EmbedderMismatchError, VectorStore


def identity_settings(**kwargs):
    return Settings(
        _env_file=None,
        runtime_embedding_provider="openai_compatible",
        runtime_embedding_base_url="https://host/v1/",
        **kwargs,
    )


def test_identity_canonical_and_sensitive_to_contract():
    from docsqa.runtime import runtime_index_identity

    settings = identity_settings()
    embedder = SimpleNamespace(
        model="m", backend="openai_compatible", dim=3, query_instruction="q"
    )
    original = runtime_index_identity(settings, embedder)
    assert original.startswith("runtime-v1:") and len(original) == 75
    assert "host" not in original
    settings.runtime_embedding_base_url = "https://host/v1/chat/completions/"
    assert runtime_index_identity(settings, embedder) == original
    for attr, value in [
        ("model", "other"),
        ("dim", 4),
        ("query_instruction", ""),
        ("backend", "other"),
    ]:
        changed = SimpleNamespace(**(vars(embedder) | {attr: value}))
        assert runtime_index_identity(settings, changed) != original
    settings.runtime_embedding_base_url = "https://other/v1"
    assert runtime_index_identity(settings, embedder) != original


@pytest.mark.parametrize("dim", [0, -1, 2001, True, 3.0])
def test_identity_rejects_invalid_dimensions(dim):
    from docsqa.runtime import RuntimeConfigurationError, runtime_index_identity

    with pytest.raises(RuntimeConfigurationError):
        runtime_index_identity(
            identity_settings(),
            SimpleNamespace(model="m", backend="api", dim=dim, query_instruction=""),
        )


class FakeConnection:
    def __init__(
        self,
        tables=(),
        existing=None,
        identity_column=True,
        fail=False,
        bad_schema=False,
        identity_column_other_schema=False,
    ):
        self.tables = set(tables)
        self.existing = existing
        self.identity_column = identity_column
        self.bad_schema = bad_schema
        self.identity_column_other_schema = identity_column_other_schema
        self.commands = []
        self.value = None
        self.rolled_back = False
        self.fail = fail
        self.in_transaction = False

    @contextmanager
    def transaction(self):
        self.in_transaction = True
        try:
            yield
        except Exception:
            self.rolled_back = True
            raise
        finally:
            self.in_transaction = False

    def execute(self, sql, params=None):
        assert self.in_transaction
        text = " ".join(sql.split())
        self.commands.append(text)
        self.value = None
        if "SELECT to_regclass('index_meta')" in text and "pg_attribute" not in text:
            self.value = tuple(
                name if name in self.tables else None
                for name in ["index_meta", "documents", "chunks"]
            )
            return self
        if "attname = 'identity'" in text:
            # new query resolves identity on the SAME relation via to_regclass
            if self.identity_column_other_schema:
                self.value = (False,)  # column lookup in wrong schema sees nothing
            else:
                self.value = (self.identity_column,)
        elif "information_schema.columns" in text:
            self.value = (self.identity_column,)
        elif "FROM index_meta WHERE" in text:
            self.value = self.existing
        elif "EXISTS (SELECT 1 FROM chunks" in text:
            self.value = (True,)
        elif "FROM pg_attribute WHERE attrelid = to_regclass('chunks')" in text:
            if self.bad_schema:
                self.value = [("chunk_id", "text")]
            else:
                self.value = [
                    ("chunk_id", "text"),
                    ("doc_id", "text"),
                    ("ordinal", "integer"),
                    ("text", "text"),
                    ("page_start", "integer"),
                    ("page_end", "integer"),
                    ("section", "text"),
                    ("embedding", "vector(3)"),
                ]
        elif "FROM pg_attribute WHERE attrelid = to_regclass('documents')" in text:
            self.value = [
                ("doc_id", "text"),
                ("source", "text"),
                ("chunk_count", "integer"),
            ]
        elif "indisprimary" in text:
            self.value = (1, 1)
        elif self.fail and text.startswith("CREATE TABLE"):
            raise RuntimeError("ddl failure")
        return self

    def fetchone(self):
        return self.value

    def fetchall(self):
        return self.value


def fake_store(conn, identity="runtime-v1:hash", dim=3):
    store = VectorStore("unused", EmbedderMeta("m", "api", dim, identity))
    store._conn = conn
    return store


@pytest.mark.parametrize(
    "tables,existing,column,identity",
    [
        (["chunks"], None, False, None),
        (["chunks"], None, False, "runtime-v1:hash"),
        (
            ["index_meta", "documents", "chunks"],
            ("m", "api", 3),
            False,
            "runtime-v1:hash",
        ),
        (
            ["index_meta", "documents", "chunks"],
            ("m", "api", 3, "runtime-v1:other"),
            True,
            "runtime-v1:hash",
        ),
        (
            ["index_meta", "documents", "chunks"],
            ("m", "api", 3, "runtime-v1:hash"),
            True,
            None,
        ),
        (["index_meta"], None, False, "runtime-v1:hash"),
    ],
)
def test_guard_refuses_before_any_mutation(tables, existing, column, identity):
    conn = FakeConnection(tables, existing, column)
    with pytest.raises(EmbedderMismatchError):
        fake_store(conn, identity).ensure_schema()
    assert not any(
        s.startswith(("CREATE", "INSERT", "ALTER", "DELETE", "UPDATE"))
        for s in conn.commands
    )
    assert conn.rolled_back


@pytest.mark.parametrize("dim", [0, -1, 2001, True])
def test_guard_rejects_dimension_before_connection(dim):
    with pytest.raises((TypeError, ValueError)):
        fake_store(FakeConnection(), dim=dim).ensure_schema()


def test_fresh_creation_is_atomic_and_locked():
    conn = FakeConnection()
    fake_store(conn).ensure_schema()
    assert "pg_advisory_xact_lock" in conn.commands[0]
    assert any(s.startswith("INSERT INTO index_meta") for s in conn.commands)
    conn = FakeConnection(fail=True)
    with pytest.raises(RuntimeError):
        fake_store(conn).ensure_schema()
    assert conn.rolled_back


def test_runtime_existing_wrong_vector_dimension_refuses_before_ddl():
    conn = FakeConnection(
        ["index_meta", "documents", "chunks"], ("m", "api", 3, "runtime-v1:hash")
    )
    with pytest.raises(EmbedderMismatchError):
        fake_store(conn).ensure_schema()
    assert not any(s.startswith(("CREATE", "INSERT", "ALTER")) for s in conn.commands)


def test_legacy_caller_cannot_use_runtime_index_when_identity_column_in_other_schema():
    # to_regclass may resolve a table in a different schema from current_schema().
    # Column lookup must use the SAME resolved relation, so a legacy caller
    # cannot be fooled into treating an identity-bearing index as legacy.
    conn = FakeConnection(
        ["index_meta", "documents", "chunks"],
        ("m", "api", 3, "runtime-v1:hash"),
        identity_column_other_schema=True,
    )
    store = VectorStore("unused", EmbedderMeta("m", "api", 3))  # no identity
    store._conn = conn
    with pytest.raises(EmbedderMismatchError):
        store.ensure_schema()
    assert not any(s.startswith(("CREATE", "INSERT")) for s in conn.commands)


def test_runtime_refuses_drifted_schema_before_ddl():
    conn = FakeConnection(
        ["index_meta", "documents", "chunks"],
        ("m", "api", 3, "runtime-v1:hash"),
        bad_schema=True,
    )
    store = VectorStore("unused", EmbedderMeta("m", "api", 3, "runtime-v1:hash"))
    store._conn = conn
    with pytest.raises(EmbedderMismatchError):
        store.ensure_schema()
    assert not any(s.startswith(("CREATE", "INSERT")) for s in conn.commands)
