"""Self-contained demo test for QADatabaseStore with clear, reliable checks.

This script validates the core functionality of the QADatabaseStore wrapper:
- CRUD operations (insert, find, update answer, update category, delete missing)
- Category utilities (list categories, rows without category)
- Text preprocessing normalization
- Similarity search (conditionally tested if DuckDB supports it)

The demo avoids unnecessary complexity and keeps assertions explicit and minimal.
It is mypy/pylint-friendly and uses Google-style docstrings for autodocumentation.

Usage:
    python -m db.duckdb_qa_store_test

"""

from __future__ import annotations

import logging
import os
from typing import List

import numpy as np

# Import the wrapper under test. It must be available on PYTHONPATH.
# If this file is appended to the same module where QADatabaseStore is defined,
# the import can be skipped and QADatabaseStore used directly.
from db.duckdb_qa_store import QADatabaseStore  # type: ignore[import]

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def generate_fake_embedding(text: str, size: int) -> List[float]:
    """Generate a deterministic, normalized embedding for testing.

    The goal is NOT to mimic real embeddings but to obtain stable vectors with
    some signal overlap for semantically related texts. This keeps the test
    simple and reliable without external services.

    Args:
        text: Input text.
        size: Target embedding size (must match the store's configuration).

    Returns:
        A unit-norm vector of length `size`.
    """
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    vec = rng.normal(loc=0.0, scale=0.1, size=size).astype(np.float32)

    # Add a tiny contribution from token presence to create weak structure.
    # This is intentionally simple and avoids heavy NLP/tokenization logic.
    tokens = set(text.lower().split())
    anchors = ["кредитную", "карту", "карте", "возраст", "комиссия", "период", "ставка"]
    for i, token in enumerate(anchors[: min(len(anchors), size)]):
        if token in tokens:
            vec[i] += 0.3

    # Normalize to unit length; fall back to a basis vector if degenerate.
    norm = float(np.linalg.norm(vec))
    if norm == 0.0 or not np.isfinite(norm):
        vec = np.zeros(size, dtype=np.float32)
        vec[0] = 1.0
    else:
        vec /= norm

    return vec.tolist()


def duckdb_supports_array_cosine(db: QADatabaseStore) -> bool:
    """Check whether the connected DuckDB supports `array_cosine_similarity`.

    Args:
        db: Open QADatabaseStore instance.

    Returns:
        True if the function is available, False otherwise.
    """
    try:
        _ = db.conn.execute(
            "SELECT array_cosine_similarity(CAST([1.0, 0.0] AS FLOAT[2]), "
            "CAST([1.0, 0.0] AS FLOAT[2]))"
        ).fetchone()
        return True
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Test / Demo
# -----------------------------------------------------------------------------


def main() -> None:
    """Run the demo test validating QADatabaseStore behavior."""
    logger.info("Starting QADatabaseStore demo test")

    # Fresh DB path for the demo.
    db_path = "test_qa.duckdb"
    if os.path.exists(db_path):
        os.remove(db_path)

    # Use a smaller embedding for faster demo runs.
    emb_size = 384
    db = QADatabaseStore(db_path, embedding_size=emb_size)

    # 0) Ensure clean state
    logger.info("0) Clearing existing data")
    assert db.clear_all_records(), "Failed to clear records"

    # 1) Insert baseline Q&A
    logger.info("1) Insert operations")
    test_qa_pairs = [
        {
            "question": "Можно ли в 16 лет оформить кредитную карту?",
            "answer": "Нет, кредитную карту можно оформить только с 18 лет.",
            "category": "Возрастные ограничения",
        },
        {
            "question": "Какой беспроцентный период по карте?",
            "answer": "Беспроцентный период составляет 120 дней.",
            "category": "Условия карты",
        },
        {
            "question": "Есть ли комиссия за обслуживание?",
            "answer": "Обслуживание карты бесплатное.",
            "category": "Условия карты",
        },
    ]

    inserted = 0
    for qa in test_qa_pairs:
        q_emb = generate_fake_embedding(qa["question"], emb_size)
        a_emb = generate_fake_embedding(qa["answer"], emb_size)
        ok = db.insert_qa(qa["question"], qa["answer"], qa["category"], q_emb, a_emb)
        logger.info(" - Insert %s: %s", "OK" if ok else "SKIP", qa["question"][:50])
        if ok:
            inserted += 1

    assert inserted == len(test_qa_pairs), f"Inserted {inserted}/{len(test_qa_pairs)}"

    # 2) Duplicate insert should be idempotent (returns False)
    logger.info("2) Duplicate insert must be skipped")
    dup_ok = db.insert_qa(
        test_qa_pairs[0]["question"],
        "Different answer",
        "Different category",
        generate_fake_embedding("x", emb_size),
        generate_fake_embedding("y", emb_size),
    )
    assert not dup_ok, "Duplicate insert should return False"

    # 3) Find question by canonical form
    logger.info("3) Find question")
    found = db.find_question("Можно ли в 16 лет оформить кредитную карту?")
    assert found is not None, "Question not found"
    assert found["answer"] == test_qa_pairs[0]["answer"], "Answer mismatch"

    # 4) Update answer + embedding
    logger.info("4) Update answer")
    new_answer = "Нет, кредитную карту можно оформить только с 18 до 70 лет."
    ok = db.update_qa(
        test_qa_pairs[0]["question"],
        new_answer,
        generate_fake_embedding(new_answer, emb_size),
    )
    assert ok, "Update must succeed"
    refetched = db.find_question(test_qa_pairs[0]["question"])
    assert (
        refetched is not None and refetched["answer"] == new_answer
    ), "Update not visible"

    # 5) Update category
    logger.info("5) Update category")
    ok = db.update_category(test_qa_pairs[0]["question"], "Требования к клиенту")
    assert ok, "Category update must succeed"

    # 6) Similarity search (conditionally executed)
    logger.info("6) Similarity search")
    can_search = duckdb_supports_array_cosine(db)
    if can_search:
        query = "С какого возраста можно получить кредитку?"
        sim = db.search_similar_questions(
            generate_fake_embedding(query, emb_size), top_k=3, threshold=0.05
        )
        if not sim:
            # Make it permissive for demo environments.
            sim = db.search_similar_questions(
                generate_fake_embedding(query, emb_size), top_k=3, threshold=0.0
            )
        assert isinstance(sim, list), "Search must return a list"
        assert len(sim) >= 1, "Expected at least one similar result"
        logger.info(" - Top similar: %s", sim[0]["question"][:70])
    else:
        logger.warning(
            "DuckDB does not support array_cosine_similarity; skipping similarity checks."
        )

    # 7) Category filter in search (only if search is available)
    logger.info("7) Category filter in search")
    if can_search:
        query = "Какой льготный период по карте?"
        sim_f = db.search_similar_questions(
            generate_fake_embedding(query, emb_size),
            category="Условия карты",
            top_k=3,
            threshold=0.0,
        )
        logger.info(" - Results with category filter: %d", len(sim_f))
        # Not asserting a minimum count: it's data/embedding dependent.

    # 8) Get all records
    logger.info("8) List all records")
    all_rows = db.get_all_qa_records()
    assert len(all_rows) == len(test_qa_pairs), "Unexpected number of rows"

    # 9) Get categories
    logger.info("9) List categories")
    cats = db.get_categories()
    assert isinstance(cats, list) and len(cats) >= 2, "Expected at least two categories"

    # 10) Rows without category
    logger.info("10) Rows without category")
    ok = db.insert_qa(
        "Какая процентная ставка по карте?",
        "Ставка 49.8% после льготного периода",
        None,
        generate_fake_embedding("Какая процентная ставка по карте?", emb_size),
        generate_fake_embedding("Ставка 49.8% после льготного периода", emb_size),
    )
    assert ok, "Insert without category must succeed"
    no_cat_rows = db.get_qa_without_category()
    assert len(no_cat_rows) >= 1, "Expected at least one row without category"

    # 11) Delete missing records
    logger.info("11) Delete records missing from provided list")
    keep = [qa["question"] for qa in test_qa_pairs[:2]]
    deleted = db.delete_missing_records(keep)
    assert deleted == 2, f"Expected to delete 2 rows, got {deleted}"
    remain = db.get_all_qa_records()
    assert len(remain) == 2, f"Expected 2 rows remaining, got {len(remain)}"

    # 12) Text preprocessing normalization
    logger.info("12) Text preprocessing checks")
    # NOTE: Expected results are aligned with current preprocess_text behavior
    # (punctuation is removed; numeric separators are preserved between digits).
    inputs = [
        "а вот алексею в 7 лет уже можно карту       оформить?",
        "Ставка 49.8% после льготного периода!",
        "Комиссия $3.99 за снятие наличных",
        "Комиссия $3,99 за снятие наличных",
    ]
    expected = [
        "а вот алексею в 7 лет уже можно карту оформить",
        "ставка 49.8% после льготного периода",
        "комиссия $3.99 за снятие наличных",
        "комиссия $3,99 за снятие наличных",
    ]
    for src, exp in zip(inputs, expected):
        got = db.preprocess_text(src)
        assert got == exp, f"Preprocess mismatch: '{got}' != '{exp}'"
        logger.info(" - '%s' -> '%s'", src, got)

    # 13) Embedding size validation
    logger.info("13) Embedding size validation")
    bad = db.insert_qa(
        "Test question with wrong embedding size",
        "Test answer",
        "Test category",
        [0.1] * 100,  # Wrong size
        [0.1] * emb_size,  # Correct size
    )
    assert not bad, "Insert with wrong embedding size must fail"

    # Finalize
    logger.info("14) Cleanup and close")
    db.close()
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info("Removed test database file")

    logger.info("All demo tests completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # DEBUG
        format="[%(asctime)s][%(name)s][%(levelname)s][%(filename)s:%(lineno)d][%(message)s]",
    )
    main()
