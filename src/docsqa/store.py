from __future__ import annotations

from dataclasses import dataclass

import psycopg

from .chunking import Chunk

# Filtered-search strategy: this deployment's image (pgvector/pgvector:pg16)
# ships pgvector 0.8.7 >= 0.8.0, so `hnsw.iterative_scan` is available and is
# used (strict_order) for doc_ids-filtered queries. This keeps HNSW candidates
# flowing past the filter so a k-candidate request over a filtered subset can
# still return k rows instead of silently truncating. If the image ever ships
# an older pgvector, the fallback is to drop the SET LOCAL below and let the
# filtered query run as an exact (seq) scan ordered by `<=>`.


@dataclass(frozen=True)
class EmbedderMeta:
    model: str
    backend: str
    dim: int
    identity: str | None = None


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    source: str
    chunk_count: int


class EmbedderMismatchError(Exception):
    """The index was built with different embedder metadata."""


_CHUNK_COLUMNS = "chunk_id, doc_id, text, page_start, page_end, section, ordinal"


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(x) for x in vector) + "]"


def _schema_matches(conn, dim: int) -> bool:
    """Return True only when existing tables match the required contract.

    Used when adopting an existing index. Anything unexpected fails closed:
    the runtime never repairs or backfills a drifted schema itself.
    """
    cols = conn.execute(
        "SELECT attname, format_type(atttypid, atttypmod) "
        "FROM pg_attribute WHERE attrelid = to_regclass('chunks') "
        "AND attnum > 0 AND NOT attisdropped"
    ).fetchall()
    chunk_columns = {name: kind for name, kind in cols}
    expected_chunk = {
        "chunk_id": "text",
        "doc_id": "text",
        "ordinal": "integer",
        "text": "text",
        "page_start": "integer",
        "page_end": "integer",
        "section": "text",
        "embedding": f"vector({dim})",
    }
    if chunk_columns != expected_chunk:
        return False
    docs = conn.execute(
        "SELECT attname, format_type(atttypid, atttypmod) "
        "FROM pg_attribute WHERE attrelid = to_regclass('documents') "
        "AND attnum > 0 AND NOT attisdropped"
    ).fetchall()
    document_columns = {name: kind for name, kind in docs}
    expected_docs = {
        "doc_id": "text",
        "source": "text",
        "chunk_count": "integer",
    }
    if document_columns != expected_docs:
        return False
    # Required primary keys must exist on the resolved relations.
    pks = conn.execute(
        "SELECT "
        "(SELECT count(*) FROM pg_index WHERE indrelid = to_regclass('chunks') "
        "AND indisprimary AND indkey[0] = "
        "(SELECT attnum FROM pg_attribute WHERE attrelid = to_regclass('chunks') "
        "AND attname = 'chunk_id')), "
        "(SELECT count(*) FROM pg_index WHERE indrelid = to_regclass('documents') "
        "AND indisprimary AND indkey[0] = "
        "(SELECT attnum FROM pg_attribute WHERE attrelid = to_regclass('documents') "
        "AND attname = 'doc_id'))"
    ).fetchone()
    return pks == (1, 1)


class VectorStore:
    """pgvector-backed chunk store with cosine-similarity search."""

    def __init__(
        self,
        database_url: str,
        embedder_meta: EmbedderMeta,
        *,
        connect_timeout: int | None = None,
        statement_timeout_ms: int | None = None,
    ) -> None:
        self.connect_options = {}
        if connect_timeout is not None:
            self.connect_options["connect_timeout"] = connect_timeout
        if statement_timeout_ms is not None:
            self.connect_options["options"] = (
                f"-c statement_timeout={int(statement_timeout_ms)}"
            )
        self.database_url = database_url
        self.embedder_meta = embedder_meta
        self._conn: psycopg.Connection | None = None

    def ensure_schema(self) -> None:
        dim = self.embedder_meta.dim
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise TypeError("embedder_meta.dim must be an int")
        if not 1 <= dim <= 2000:
            raise ValueError("embedding dimension requires a compatible fresh index")
        conn = self._connection()
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(731924610)")
            tables = conn.execute(
                "SELECT to_regclass('index_meta'), to_regclass('documents'), to_regclass('chunks')"
            ).fetchone()
            meta_table, _documents_table, chunks_table = tables
            expected = self.embedder_meta
            existing = None
            identity_column = False
            if meta_table:
                # Resolve the identity column on the SAME relation used by
                # the metadata SELECT below. `current_schema()` can differ
                # from the schema that `to_regclass`/`FROM index_meta` used.
                identity_column = conn.execute(
                    "SELECT EXISTS (SELECT 1 FROM pg_attribute "
                    "WHERE attrelid = to_regclass('index_meta') "
                    "AND attname = 'identity' AND NOT attisdropped)"
                ).fetchone()[0]
                columns = "model, backend, dim" + (
                    ", identity" if identity_column else ""
                )
                existing = conn.execute(
                    f"SELECT {columns} FROM index_meta WHERE id = 1"
                ).fetchone()
            if (
                expected.identity is not None
                and any(tables)
                and (not all(tables) or not identity_column or existing is None)
            ):
                raise EmbedderMismatchError("fresh index/rebuild required")
            if existing is not None:
                actual = tuple(existing) if identity_column else (*existing, None)
                if actual != (
                    expected.model,
                    expected.backend,
                    expected.dim,
                    expected.identity,
                ):
                    raise EmbedderMismatchError("fresh index/rebuild required")
                if expected.identity is not None and not _schema_matches(conn, dim):
                    raise EmbedderMismatchError("fresh index/rebuild required")
            elif (
                chunks_table
                and conn.execute(
                    "SELECT EXISTS (SELECT 1 FROM chunks LIMIT 1)"
                ).fetchone()[0]
            ):
                raise EmbedderMismatchError("fresh index/rebuild required")
            # All identity checks precede extension/table/metadata mutation.
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS index_meta (
                    id integer PRIMARY KEY, model text NOT NULL,
                    backend text NOT NULL, dim integer NOT NULL, identity text
                )
            """)
            if existing is None:
                if not meta_table or identity_column:
                    conn.execute(
                        "INSERT INTO index_meta (id, model, backend, dim, identity) "
                        "VALUES (1, %s, %s, %s, %s)",
                        (expected.model, expected.backend, dim, expected.identity),
                    )
                else:
                    conn.execute(
                        "INSERT INTO index_meta (id, model, backend, dim) VALUES (1, %s, %s, %s)",
                        (expected.model, expected.backend, dim),
                    )
            # DDL is idempotent; vector(dim) can only be interpolated because PG
            # rejects typmod parameters — dim is int-validated above.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id text PRIMARY KEY,
                    source text NOT NULL,
                    chunk_count integer NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id text PRIMARY KEY,
                    doc_id text NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
                    ordinal integer NOT NULL,
                    text text NOT NULL,
                    page_start integer NOT NULL,
                    page_end integer NOT NULL,
                    section text,
                    embedding vector({dim}) NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )

    def upsert_document(
        self,
        doc_id: str,
        source: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for vector in vectors:
            if len(vector) != self.embedder_meta.dim:
                raise ValueError(
                    f"vector dim {len(vector)} does not match index dim "
                    f"{self.embedder_meta.dim}"
                )
        conn = self._connection()
        with conn.transaction():
            conn.execute(
                """
                INSERT INTO documents (doc_id, source, chunk_count)
                VALUES (%s, %s, %s)
                ON CONFLICT (doc_id)
                DO UPDATE SET source = EXCLUDED.source, chunk_count = EXCLUDED.chunk_count
                """,
                (doc_id, source, len(chunks)),
            )
            # Delete-and-reinsert keeps a re-upsert idempotent even when the
            # chunk count shrank; chunk_ids are deterministic (doc_id:ordinal).
            conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            with conn.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO chunks
                        (chunk_id, doc_id, ordinal, text, page_start, page_end, section, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    [
                        (
                            chunk.chunk_id,
                            chunk.doc_id,
                            chunk.ordinal,
                            chunk.text,
                            chunk.page_start,
                            chunk.page_end,
                            chunk.section,
                            _vector_literal(vector),
                        )
                        for chunk, vector in zip(chunks, vectors, strict=True)
                    ],
                )
        return len(chunks)

    def search_vector(
        self,
        qvec: list[float],
        k: int,
        doc_ids: list[str] | None = None,
    ) -> list[Hit]:
        params: dict[str, object] = {
            "q": _vector_literal(qvec),
            "k": k,
        }
        where = ""
        if doc_ids is not None:
            where = "WHERE doc_id = ANY(%(doc_ids)s)"
            params["doc_ids"] = doc_ids
        query = f"""
            SELECT {_CHUNK_COLUMNS},
                   1 - (embedding <=> %(q)s::vector) AS score
            FROM chunks
            {where}
            ORDER BY embedding <=> %(q)s::vector
            LIMIT %(k)s
        """
        conn = self._connection()
        with conn.transaction():
            if doc_ids is not None:
                conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
            rows = conn.execute(query, params).fetchall()
        return [
            Hit(
                chunk=Chunk(
                    chunk_id=row[0],
                    doc_id=row[1],
                    text=row[2],
                    page_start=row[3],
                    page_end=row[4],
                    section=row[5],
                    ordinal=row[6],
                ),
                score=row[7],
            )
            for row in rows
        ]

    def all_chunks(self, doc_ids: list[str] | None = None) -> list[Chunk]:
        params: dict[str, object] = {}
        where = ""
        if doc_ids is not None:
            where = "WHERE doc_id = ANY(%(doc_ids)s)"
            params["doc_ids"] = doc_ids
        conn = self._connection()
        rows = conn.execute(
            f"""
            SELECT {_CHUNK_COLUMNS}
            FROM chunks
            {where}
            ORDER BY doc_id, ordinal
            """,
            params,
        ).fetchall()
        conn.commit()
        return [
            Chunk(
                chunk_id=row[0],
                doc_id=row[1],
                text=row[2],
                page_start=row[3],
                page_end=row[4],
                section=row[5],
                ordinal=row[6],
            )
            for row in rows
        ]

    def list_documents(self) -> list[DocumentInfo]:
        conn = self._connection()
        rows = conn.execute(
            "SELECT doc_id, source, chunk_count FROM documents ORDER BY doc_id"
        ).fetchall()
        conn.commit()
        return [DocumentInfo(*row) for row in rows]

    def delete_document(self, doc_id: str) -> None:
        conn = self._connection()
        with conn.transaction():
            conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))

    def check_ready(self) -> bool:
        # Dedicated short-lived connection: readiness must not join a request
        # transaction or keep reusing a connection broken by a DB restart.
        conn = None
        try:
            conn = psycopg.connect(self.database_url, **self.connect_options)
            return conn.execute("SELECT 1").fetchone() == (1,)
        except psycopg.Error:
            return False
        finally:
            if conn is not None:
                conn.close()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _connection(self) -> psycopg.Connection:
        if self._conn is None:
            self._conn = psycopg.connect(self.database_url, **self.connect_options)
        return self._conn
