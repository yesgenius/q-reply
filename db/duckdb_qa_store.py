"""DuckDB wrapper for a Q&A store with vector embeddings (simple & reliable).

This module provides a minimal, maintainable wrapper around DuckDB for storing
question–answer pairs with fixed-size float embeddings and performing
cosine-similarity search **inside SQL**.

Design goals:
* Simple, explicit, mypy/pylint-friendly.
* No premature optimization or unnecessary abstraction.
* Clear validation and consistent transactions.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

import duckdb
import numpy as np


# ---------------------------------------------------------------------------
# Public constants (keep small, explicit surface)
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_SIZE = 2560
DEFAULT_SIMILARITY_THRESHOLD = 0.15
DEFAULT_TOP_K = 5

logger = logging.getLogger(__name__)


class QADatabaseStore:
    """DuckDB-backed store for Q&A with embeddings.

    The table uses fixed-size float arrays (FLOAT[N]) for embeddings. All
    embeddings are validated (length, finiteness, non-zero norm) and stored as
    normalized vectors.

    Attributes:
        db_path: Path to the DuckDB file.
        embedding_size: Expected dimensionality for embeddings.
        conn: Open DuckDB connection.
    """

    # ----------------------------- Lifecycle --------------------------------

    def __init__(
        self, db_path: str, embedding_size: int = DEFAULT_EMBEDDING_SIZE
    ) -> None:
        """Create the store and initialize schema.

        Args:
            db_path: Path to the DuckDB database file.
            embedding_size: Fixed embedding dimensionality for this DB.

        Raises:
            RuntimeError: When schema exists with a different embedding size.
            duckdb.Error: On connection or DDL failures.
        """
        self.db_path = db_path
        self.embedding_size = int(embedding_size)
        self.conn: duckdb.DuckDBPyConnection = duckdb.connect(self.db_path)

        self._initialize_database()

    # --------------------------- Schema management ---------------------------

    def _initialize_database(self) -> None:
        """Create or validate schema.

        Creates the table if it does not exist. If it exists, verifies the
        embedding column types match the configured `embedding_size`.

        Raises:
            RuntimeError: If an existing schema has different embedding sizes.
        """
        if self._table_exists("qa_records"):
            self._validate_schema_or_raise()
            return

        # Fresh create: fixed-size embedding arrays (FLOAT[N]).
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS qa_id_seq START 1")
        self.conn.execute(
            f"""
            CREATE TABLE qa_records (
                id BIGINT PRIMARY KEY DEFAULT nextval('qa_id_seq'),
                question TEXT UNIQUE NOT NULL,
                answer TEXT NOT NULL,
                category TEXT,
                question_embedding FLOAT[{self.embedding_size}],
                answer_embedding  FLOAT[{self.embedding_size}],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_category ON qa_records(category)"
        )

    def _table_exists(self, name: str) -> bool:
        """Check whether a table exists.

        Args:
            name: Table name.

        Returns:
            True if the table exists, else False.
        """
        row = self.conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ? LIMIT 1",
            [name],
        ).fetchone()
        return row is not None

    def _validate_schema_or_raise(self) -> None:
        """Validate that embedding columns match the configured size.

        Raises:
            RuntimeError: If types do not match `FLOAT[embedding_size]`.
        """
        info = self.conn.execute("PRAGMA table_info('qa_records')").fetchall()
        # table_info: [cid, name, type, notnull, dflt_value, pk]
        types: dict[str, str] = {str(r[1]).lower(): str(r[2]).upper() for r in info}

        expected = f"FLOAT[{self.embedding_size}]"
        q_type = types.get("question_embedding", "")
        a_type = types.get("answer_embedding", "")
        if q_type != expected or a_type != expected:
            raise RuntimeError(
                f"Embedding size mismatch: DB has question={q_type} answer={a_type}, "
                f"expected {expected}. Recreate the database or align embedding_size."
            )

    # ----------------------------- Sanitization ------------------------------

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Normalize input text to a canonical form.

        This removes most punctuation (keeps numeric patterns like 49.8%, $3.99),
        squashes whitespace, and lowercases.

        Args:
            text: Raw text.

        Returns:
            Canonicalized text string.
        """
        pattern = r"""
            (?<!\d)[.,;:!?'"`~@#^&*\-_=+[\]{}/\\|<>]|   # punctuation not after digit
            [.,;:!?'"`~@#^&*\-_=+[\]{}/\\|<>](?!\d)     # punctuation not before digit
            (?!(?<=\d)[%$])                             # keep % or $ after digits
        """
        cleaned = re.sub(pattern, "", text, flags=re.VERBOSE)
        return re.sub(r"\s+", " ", cleaned.lower()).strip()

    @staticmethod
    def _normalize_category(value: str | None) -> str | None:
        """Trim category input; map empty/whitespace-only to None.

        Args:
            value: Category string or None.

        Returns:
            Normalized category or None.
        """
        if value is None:
            return None
        normalized = value.strip()
        return normalized if normalized else None

    # -------------------------- Embedding validation -------------------------

    def _validate_and_normalize_embedding(self, emb: Sequence[float]) -> list[float]:
        """Validate and L2-normalize an embedding.

        Args:
            emb: Embedding sequence to validate.

        Returns:
            Normalized embedding list (float32 precision).

        Raises:
            ValueError: If size mismatch, non-finite values, or zero norm.
        """
        if len(emb) != self.embedding_size:
            raise ValueError(
                f"Embedding size mismatch: expected {self.embedding_size}, got {len(emb)}"
            )

        arr = np.asarray(emb, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError("Embedding contains non-finite values (NaN/Inf).")

        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            raise ValueError("Zero-norm embedding is not allowed.")

        return (arr / norm).tolist()

    # ------------------------------- Queries ---------------------------------

    def find_question(self, question: str) -> dict[str, Any] | None:
        """Find an exact question by its canonical form.

        Args:
            question: Original question text.

        Returns:
            Record dict if found; otherwise None.
        """
        q = self.preprocess_text(question)
        row = self.conn.execute(
            "SELECT * FROM qa_records WHERE question = ?", [q]
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "question": row[1],
            "answer": row[2],
            "category": row[3],
            "question_embedding": row[4],
            "answer_embedding": row[5],
            "created_at": row[6],
            "updated_at": row[7],
        }

    # ------------------------------- Mutations -------------------------------

    def insert_qa(
        self,
        question: str,
        answer: str,
        category: str | None,
        question_embedding: Sequence[float],
        answer_embedding: Sequence[float],
    ) -> bool:
        """Insert a Q&A row (idempotent on question text).

        Args:
            question: Question text (will be canonicalized).
            answer: Answer text.
            category: Optional category (whitespace-only will be stored as NULL).
            question_embedding: Question embedding.
            answer_embedding: Answer embedding.

        Returns:
            True on success; False on any validation/constraint error.
        """
        q = self.preprocess_text(question)
        cat = self._normalize_category(category)

        try:
            q_emb = self._validate_and_normalize_embedding(question_embedding)
            a_emb = self._validate_and_normalize_embedding(answer_embedding)
        except ValueError as exc:
            logger.error("Insert failed: %s", exc)
            return False

        try:
            self.conn.execute("BEGIN")
            # Pass embeddings as lists; let DuckDB cast to fixed-size FLOAT[N].
            self.conn.execute(
                f"""
                INSERT INTO qa_records
                (question, answer, category, question_embedding, answer_embedding)
                VALUES (?, ?, ?, CAST(? AS FLOAT[{self.embedding_size}]),
                              CAST(? AS FLOAT[{self.embedding_size}]))
                """,
                [q, answer, cat, q_emb, a_emb],
            )
            self.conn.execute("COMMIT")
            return True
        except duckdb.ConstraintException as exc:
            # Duplicate unique(question) → not an error for idempotent insert.
            self.conn.execute("ROLLBACK")
            msg = str(exc).upper()
            if "UNIQUE" in msg or "DUPLICATE" in msg:
                logger.info("Insert skipped: question already exists: %s", q[:80])
                return False
            logger.error("Insert constraint error: %s", exc)
            return False
        except Exception as exc:  # pragma: no cover - conservative safety
            try:
                self.conn.execute("ROLLBACK")
            finally:
                logger.error("Insert failed: %s", exc)
            return False

    def update_qa(
        self, question: str, answer: str, answer_embedding: Sequence[float]
    ) -> bool:
        """Update answer and answer embedding for an existing question.

        Args:
            question: Original question text.
            answer: New answer text.
            answer_embedding: New answer embedding.

        Returns:
            True if a row was updated; False if not found or on validation error.
        """
        q = self.preprocess_text(question)
        try:
            a_emb = self._validate_and_normalize_embedding(answer_embedding)
        except ValueError as exc:
            logger.error("Update failed: %s", exc)
            return False

        try:
            self.conn.execute("BEGIN")
            self.conn.execute(
                f"""
                UPDATE qa_records
                SET answer = ?, 
                    answer_embedding = CAST(? AS FLOAT[{self.embedding_size}]),
                    updated_at = CURRENT_TIMESTAMP
                WHERE question = ?
                """,
                [answer, a_emb, q],
            )
            # Verify presence (simple and explicit)
            row = self.conn.execute(
                "SELECT COUNT(*) FROM qa_records WHERE question = ?", [q]
            ).fetchone()
            found = bool(row and row[0] > 0)
            if found:
                self.conn.execute("COMMIT")
                return True
            self.conn.execute("ROLLBACK")
            logger.warning("Update skipped: question not found: %s", q[:80])
            return False
        except Exception as exc:  # pragma: no cover
            try:
                self.conn.execute("ROLLBACK")
            finally:
                logger.error("Update failed: %s", exc)
            return False

    def update_category(self, question: str, category: str | None) -> bool:
        """Update category for an existing question.

        Args:
            question: Original question text.
            category: New category (whitespace-only becomes NULL).

        Returns:
            True if a row was updated; False if not found.
        """
        q = self.preprocess_text(question)
        cat = self._normalize_category(category)

        try:
            self.conn.execute("BEGIN")
            self.conn.execute(
                "UPDATE qa_records SET category = ?, updated_at = CURRENT_TIMESTAMP WHERE question = ?",
                [cat, q],
            )
            row = self.conn.execute(
                "SELECT COUNT(*) FROM qa_records WHERE question = ?", [q]
            ).fetchone()
            found = bool(row and row[0] > 0)
            if found:
                self.conn.execute("COMMIT")
                return True
            self.conn.execute("ROLLBACK")
            logger.warning("Category update skipped: question not found: %s", q[:80])
            return False
        except Exception as exc:  # pragma: no cover
            try:
                self.conn.execute("ROLLBACK")
            finally:
                logger.error("Category update failed: %s", exc)
            return False

    # ---------------------------- Similarity search --------------------------

    def search_similar_questions(
        self,
        question_embedding: Sequence[float],
        category: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> list[dict[str, Any]]:
        """Search similar questions using SQL `array_cosine_similarity`.

        The input embedding is validated and normalized. The SQL casts the
        parameter list into the fixed-size FLOAT[N] array type to align with the
        column type, then uses `array_cosine_similarity` in DuckDB.

        Args:
            question_embedding: Query embedding.
            category: Optional category filter (whitespace-only is ignored).
            top_k: Maximum number of results returned.
            threshold: Minimal cosine similarity.

        Returns:
            Sorted list of result dicts: id, question, answer, category, similarity.
        """
        try:
            q_emb = self._validate_and_normalize_embedding(question_embedding)
        except ValueError as exc:
            logger.error("Search failed: %s", exc)
            return []

        cat = self._normalize_category(category)

        try:
            if cat is not None:
                sql = f"""
                    WITH scored AS (
                        SELECT
                            id, question, answer, category,
                            array_cosine_similarity(
                                question_embedding,
                                CAST(? AS FLOAT[{self.embedding_size}])
                            ) AS similarity
                        FROM qa_records
                        WHERE category = ?
                    )
                    SELECT id, question, answer, category, similarity
                    FROM scored
                    WHERE similarity >= ?
                    ORDER BY similarity DESC
                    LIMIT ?
                """
                rows = self.conn.execute(
                    sql, [q_emb, cat, float(threshold), int(top_k)]
                ).fetchall()
            else:
                sql = f"""
                    WITH scored AS (
                        SELECT
                            id, question, answer, category,
                            array_cosine_similarity(
                                question_embedding,
                                CAST(? AS FLOAT[{self.embedding_size}])
                            ) AS similarity
                        FROM qa_records
                    )
                    SELECT id, question, answer, category, similarity
                    FROM scored
                    WHERE similarity >= ?
                    ORDER BY similarity DESC
                    LIMIT ?
                """
                rows = self.conn.execute(
                    sql, [q_emb, float(threshold), int(top_k)]
                ).fetchall()

            # Optional light deduplication by answer text (keep highest sim).
            best_by_answer: dict[str, tuple[int, str, str, str | None, float]] = {}
            for rid, qtext, ans, catval, sim in rows:
                prev = best_by_answer.get(ans)
                if prev is None or sim > prev[4]:
                    best_by_answer[ans] = (rid, qtext, ans, catval, sim)

            results = [
                {
                    "id": v[0],
                    "question": v[1],
                    "answer": v[2],
                    "category": v[3],
                    "similarity": float(v[4]),
                }
                for v in sorted(
                    best_by_answer.values(), key=lambda t: t[4], reverse=True
                )[: int(top_k)]
            ]
            return results
        except Exception as exc:  # pragma: no cover
            logger.error("Search failed: %s", exc)
            return []

    # ------------------------------ Utilities --------------------------------

    def get_all_qa_records(self) -> list[dict[str, Any]]:
        """Return all Q&A records (lightweight listing)."""
        rows = self.conn.execute(
            "SELECT id, question, answer, category, created_at, updated_at FROM qa_records ORDER BY id"
        ).fetchall()
        return [
            {
                "id": r[0],
                "question": r[1],
                "answer": r[2],
                "category": r[3],
                "created_at": r[4],
                "updated_at": r[5],
            }
            for r in rows
        ]

    def get_categories(self) -> list[str]:
        """Return all distinct non-null categories sorted alphabetically."""
        rows = self.conn.execute(
            "SELECT DISTINCT category FROM qa_records WHERE category IS NOT NULL ORDER BY category"
        ).fetchall()
        return [r[0] for r in rows]

    def get_distinct_categories_from_qa(self) -> list[str]:
        """Alias for get_categories() (clarity)."""
        return self.get_categories()

    def get_qa_without_category(self) -> list[dict[str, Any]]:
        """Return Q&A records with NULL or empty category."""
        rows = self.conn.execute(
            "SELECT id, question, answer, category, created_at, updated_at "
            "FROM qa_records WHERE category IS NULL OR category = '' ORDER BY id"
        ).fetchall()
        return [
            {
                "id": r[0],
                "question": r[1],
                "answer": r[2],
                "category": r[3],
                "created_at": r[4],
                "updated_at": r[5],
            }
            for r in rows
        ]

    def delete_missing_records(self, current_questions: Sequence[str]) -> int:
        """Delete rows whose `question` is not in the provided list.

        Args:
            current_questions: Questions to keep (any others will be deleted).

        Returns:
            Count of deleted rows.
        """
        keep = [self.preprocess_text(q) for q in current_questions]

        try:
            self.conn.execute("BEGIN")
            before = self.conn.execute("SELECT COUNT(*) FROM qa_records").fetchone()
            total_before = int(before[0]) if before else 0

            if keep:
                placeholders = ",".join(["?"] * len(keep))
                self.conn.execute(
                    f"DELETE FROM qa_records WHERE question NOT IN ({placeholders})",
                    list(keep),
                )
            else:
                self.conn.execute("DELETE FROM qa_records")

            after = self.conn.execute("SELECT COUNT(*) FROM qa_records").fetchone()
            total_after = int(after[0]) if after else 0
            deleted = total_before - total_after

            self.conn.execute("COMMIT")
            return deleted
        except Exception as exc:  # pragma: no cover
            try:
                self.conn.execute("ROLLBACK")
            finally:
                logger.error("Deletion failed: %s", exc)
            return 0

    def clear_all_records(self) -> bool:
        """Remove all rows (useful for tests)."""
        try:
            self.conn.execute("DELETE FROM qa_records")
            return True
        except Exception as exc:  # pragma: no cover
            logger.error("Clear failed: %s", exc)
            return False

    def close(self) -> None:
        """Close the DuckDB connection."""
        try:
            self.conn.close()
        finally:
            # Make it idempotent for repeated close() in callers.
            self.conn = duckdb.connect(self.db_path)
            self.conn.close()
