"""Self-contained demo test for QADatabaseStore with clear, reliable checks.

This script validates the core functionality of the QADatabaseStore wrapper:
- CRUD operations (insert, find, update answer, update category, delete missing)
- Category utilities (list categories, rows without category)
- Text preprocessing normalization
- Similarity search with comprehensive parameter validation
- Embedding size validation
- Case-insensitive category handling
- Original question text preservation

The demo avoids unnecessary complexity and keeps assertions explicit and minimal.
It is mypy/pylint-friendly and uses Google-style docstrings for autodocumentation.

Usage:
    python -m storages.duckdb_qa_store_test

"""

from __future__ import annotations

import logging
import os
from typing import cast

import numpy as np

# Import the wrapper under test. It must be available on PYTHONPATH.
# If this file is appended to the same module where QADatabaseStore is defined,
# the import can be skipped and QADatabaseStore used directly.
from storages.duckdb_qa_store import QADatabaseStore  # type: ignore[import]


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def generate_fake_embedding(text: str, size: int) -> list[float]:
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

    return cast("list[float]", vec.tolist())


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

    # 3.1) NEW: Verify original question text is preserved
    logger.info("3.1) Verify original question text preservation and normalization")
    # Insert question with specific punctuation and casing
    test_question_original = "What's the BEST credit-card for 2025?!"
    test_answer = "Our premium card offers the best rewards."
    ok = db.insert_qa(
        test_question_original,
        test_answer,
        "Cards",
        generate_fake_embedding(test_question_original, emb_size),
        generate_fake_embedding(test_answer, emb_size),
    )
    assert ok, "Failed to insert test question"

    # Test that ALL different variants find the SAME record
    # This verifies normalization actually works for deduplication
    test_variants = [
        ("What's the BEST credit-card for 2025?!", "original formatting"),
        ("whats the best creditcard for 2025", "fully normalized"),
        ("WHAT'S THE BEST CREDIT-CARD FOR 2025", "uppercase variant"),
        ("what's   the  BEST  credit-card  for  2025?", "extra spaces variant"),
        ("What's the best credit-card for 2025", "mixed case no punctuation"),
    ]

    logger.info("  - Testing normalization with %d variants", len(test_variants))
    for variant, description in test_variants:
        found_variant = db.find_question(variant)
        assert found_variant is not None, f"Failed to find with {description}: '{variant}'"
        # All variants should return the same original text
        assert found_variant["question"] == test_question_original.strip(), (
            f"Original text not preserved for {description}: "
            f"expected '{test_question_original.strip()}', got '{found_variant['question']}'"
        )
        assert found_variant["answer"] == test_answer, f"Wrong answer for {description}"
        logger.info("    - ✓ Found with %s", description)

    # Also verify the original Russian question maintains its formatting
    found_russian = db.find_question("Можно ли в 16 лет оформить кредитную карту?")
    assert found_russian["question"] == "Можно ли в 16 лет оформить кредитную карту?", (
        "Original Russian question text not preserved"
    )

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
    assert refetched is not None and refetched["answer"] == new_answer, "Update not visible"

    # 5) Update category
    logger.info("5) Update category")
    ok = db.update_category(test_qa_pairs[0]["question"], "Требования к клиенту")
    assert ok, "Category update must succeed"

    # 6) Similarity search - comprehensive tests
    logger.info("6) Similarity search - comprehensive parameter validation")
    can_search = duckdb_supports_array_cosine(db)
    if can_search:
        # Prepare more test data for thorough testing
        logger.info(" - Adding more test data for search validation")
        additional_qa = [
            (
                "С какого возраста выдают кредитку?",
                "С 18 лет можно оформить.",
                "Возрастные ограничения",
            ),
            (
                "Минимальный возраст для кредитной карты?",
                "18 лет минимум.",
                "Возрастные ограничения",
            ),
            (
                "Какая процентная ставка по карте?",
                "Ставка 24.9% годовых.",
                "Условия карты",
            ),
            (
                "Есть ли льготный период?",
                "Да, 120 дней без процентов.",
                "Условия карты",
            ),
            ("Какие документы нужны?", "Паспорт и СНИЛС.", "Документы"),
            ("Нужна ли справка о доходах?", "Нет, справка не требуется.", "Документы"),
            ("Можно ли снимать наличные?", "Да, но с комиссией 3%.", "Операции"),
        ]

        for q, a, cat in additional_qa:
            db.insert_qa(
                q,
                a,
                cat,
                generate_fake_embedding(q, emb_size),
                generate_fake_embedding(a, emb_size),
            )

        # Test 6.1: Validate threshold impact
        logger.info(" 6.1) Testing threshold impact on result count")
        query_emb = generate_fake_embedding("возраст для получения кредитной карты", emb_size)

        results_high_threshold = db.search_similar_questions(query_emb, top_k=20, threshold=0.8)
        results_mid_threshold = db.search_similar_questions(query_emb, top_k=20, threshold=0.3)
        results_low_threshold = db.search_similar_questions(query_emb, top_k=20, threshold=0.0)

        # Higher threshold should return fewer or equal results
        assert len(results_high_threshold) <= len(results_mid_threshold), (
            f"High threshold ({len(results_high_threshold)}) should yield fewer results than mid ({len(results_mid_threshold)})"
        )
        assert len(results_mid_threshold) <= len(results_low_threshold), (
            f"Mid threshold ({len(results_mid_threshold)}) should yield fewer results than low ({len(results_low_threshold)})"
        )

        # Verify all results meet threshold requirement
        for r in results_high_threshold:
            assert r["similarity"] >= 0.8, (
                f"Result similarity {r['similarity']} below threshold 0.8"
            )
        for r in results_mid_threshold:
            assert r["similarity"] >= 0.3, (
                f"Result similarity {r['similarity']} below threshold 0.3"
            )

        logger.info("  - Threshold 0.8: %d results", len(results_high_threshold))
        logger.info("  - Threshold 0.3: %d results", len(results_mid_threshold))
        logger.info("  - Threshold 0.0: %d results", len(results_low_threshold))

        # Test 6.2: Validate top_k impact
        logger.info(" 6.2) Testing top_k impact on result count")

        results_k1 = db.search_similar_questions(query_emb, top_k=1, threshold=0.0)
        results_k3 = db.search_similar_questions(query_emb, top_k=3, threshold=0.0)
        results_k10 = db.search_similar_questions(query_emb, top_k=10, threshold=0.0)
        results_k50 = db.search_similar_questions(query_emb, top_k=50, threshold=0.0)

        # Verify top_k constraint
        assert len(results_k1) <= 1, f"top_k=1 returned {len(results_k1)} results"
        assert len(results_k3) <= 3, f"top_k=3 returned {len(results_k3)} results"
        assert len(results_k10) <= 10, f"top_k=10 returned {len(results_k10)} results"
        assert len(results_k50) <= 50, f"top_k=50 returned {len(results_k50)} results"

        # Results should increase or stay same as k increases
        assert len(results_k1) <= len(results_k3), "k=1 should have <= results than k=3"
        assert len(results_k3) <= len(results_k10), "k=3 should have <= results than k=10"
        assert len(results_k10) <= len(results_k50), "k=10 should have <= results than k=50"

        logger.info("  - top_k=1: %d results", len(results_k1))
        logger.info("  - top_k=3: %d results", len(results_k3))
        logger.info("  - top_k=10: %d results", len(results_k10))
        logger.info("  - top_k=50: %d results", len(results_k50))

        # Test 6.3: Validate category filter impact
        logger.info(" 6.3) Testing category filter impact on result count")

        # Search without category filter
        results_no_filter = db.search_similar_questions(query_emb, top_k=20, threshold=0.0)

        # Search with specific category filters
        results_age_cat = db.search_similar_questions(
            query_emb, category="Возрастные ограничения", top_k=20, threshold=0.0
        )
        results_terms_cat = db.search_similar_questions(
            query_emb, category="Условия карты", top_k=20, threshold=0.0
        )
        results_docs_cat = db.search_similar_questions(
            query_emb, category="Документы", top_k=20, threshold=0.0
        )

        # Category filter should reduce or maintain result count
        assert len(results_age_cat) <= len(results_no_filter), (
            f"Filtered results ({len(results_age_cat)}) exceed unfiltered ({len(results_no_filter)})"
        )
        assert len(results_terms_cat) <= len(results_no_filter), (
            f"Filtered results ({len(results_terms_cat)}) exceed unfiltered ({len(results_no_filter)})"
        )

        # Verify all filtered results have correct category (normalized to lowercase)
        for r in results_age_cat:
            assert r["category"] == "возрастные ограничения", (
                f"Result has wrong category: {r['category']}"
            )
        for r in results_terms_cat:
            assert r["category"] == "условия карты", f"Result has wrong category: {r['category']}"

        logger.info("  - No filter: %d results", len(results_no_filter))
        logger.info("  - Category 'Возрастные ограничения': %d results", len(results_age_cat))
        logger.info("  - Category 'Условия карты': %d results", len(results_terms_cat))
        logger.info("  - Category 'Документы': %d results", len(results_docs_cat))

        # Test 6.4: Validate result ordering by similarity
        logger.info(" 6.4) Testing result ordering by similarity")
        results = db.search_similar_questions(query_emb, top_k=10, threshold=0.0)

        if len(results) > 1:
            for i in range(1, len(results)):
                assert results[i - 1]["similarity"] >= results[i]["similarity"], (
                    f"Results not ordered: {results[i - 1]['similarity']} < {results[i]['similarity']}"
                )
            logger.info("  - Results correctly ordered by descending similarity")

        # Test 6.5: Edge cases
        logger.info(" 6.5) Testing edge cases")

        # Very high threshold should return few/no results
        results_impossible = db.search_similar_questions(query_emb, top_k=10, threshold=0.99)
        logger.info(
            "  - Threshold 0.99: %d results (expected 0 or very few)",
            len(results_impossible),
        )

        # Non-existent category should return empty
        results_no_cat = db.search_similar_questions(
            query_emb, category="NonExistentCategory", top_k=10, threshold=0.0
        )
        assert len(results_no_cat) == 0, "Non-existent category should return no results"
        logger.info("  - Non-existent category: 0 results (as expected)")

        # 6.6) NEW: Verify search results preserve original text
        logger.info(" 6.6) Verify search results preserve original question text")
        if len(results_low_threshold) > 0:
            # Check that at least one result has punctuation (if it was in original)
            has_punctuation = any("?" in r["question"] for r in results_low_threshold)
            logger.info("  - Search results contain original punctuation: %s", has_punctuation)

            # Specifically check for our test question
            test_q_results = db.search_similar_questions(
                generate_fake_embedding("credit card 2025", emb_size), top_k=10, threshold=0.0
            )
            for r in test_q_results:
                if "2025" in r["question"]:
                    assert "What's the BEST credit-card for 2025?!" in r["question"], (
                        f"Original formatting not preserved in search: {r['question']}"
                    )
                    logger.info("  - Found original formatted text in search: %s", r["question"])
                    break

    else:
        logger.warning(
            "DuckDB does not support array_cosine_similarity; skipping similarity checks."
        )

    # 7) Get all records
    logger.info("7) List all records")
    all_rows = db.get_all_qa_records()
    # We added more records in test 6
    assert len(all_rows) >= len(test_qa_pairs), "Unexpected number of rows"

    # 7.1) NEW: Verify all records return original text
    logger.info("7.1) Verify all records return original text")
    for row in all_rows:
        if "2025" in row["question"]:
            assert "What's the BEST credit-card for 2025?!" in row["question"], (
                f"Original text not preserved in listing: {row['question']}"
            )
            logger.info("  - All records preserve original text: %s", row["question"][:50])
            break

    # 8) Get categories
    logger.info("8) List categories")
    cats = db.get_categories()
    assert isinstance(cats, list) and len(cats) >= 2, "Expected at least two categories"

    # 9) Rows without category
    logger.info("9) Rows without category")
    ok = db.insert_qa(
        "Можно ли пополнить карту без комиссии?",
        "Да, пополнение без комиссии через банкоматы банка",
        None,
        generate_fake_embedding("Можно ли пополнить карту без комиссии?", emb_size),
        generate_fake_embedding("Да, пополнение без комиссии через банкоматы банка", emb_size),
    )
    assert ok, "Insert without category must succeed"
    logger.info(" - Inserted row without category")

    no_cat_rows = db.get_qa_without_category()
    assert len(no_cat_rows) >= 1, "Expected at least one row without category"

    # 9.1) NEW: Verify rows without category preserve original text
    logger.info("9.1) Verify rows without category preserve original text")
    for row in no_cat_rows:
        if "пополнить" in row["question"]:
            assert row["question"] == "Можно ли пополнить карту без комиссии?", (
                f"Original text not preserved in no-category listing: {row['question']}"
            )
            logger.info("  - No-category rows preserve original text")
            break

    # 10) Delete missing records
    logger.info("10) Delete records missing from provided list")
    # Get current count before deletion
    before_delete = len(db.get_all_qa_records())

    # Keep only the first two original questions
    keep = [qa["question"] for qa in test_qa_pairs[:2]]
    deleted = db.delete_missing_records(keep)

    remain = db.get_all_qa_records()
    assert len(remain) == 2, f"Expected 2 rows remaining, got {len(remain)}"
    logger.info(" - Deleted %d rows, %d remaining", deleted, len(remain))

    # 11) Text preprocessing normalization
    logger.info("11) Text preprocessing checks")
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
    for src, exp in zip(inputs, expected, strict=False):
        got = db.preprocess_text(src)
        assert got == exp, f"Preprocess mismatch: '{got}' != '{exp}'"
        logger.info(" - '%s' -> '%s'", src, got)

    # 12) Embedding size validation
    logger.info("12) Embedding size validation")
    bad = db.insert_qa(
        "Test question with wrong embedding size",
        "Test answer",
        "Test category",
        [0.1] * 100,  # Wrong size
        [0.1] * emb_size,  # Correct size
    )
    assert not bad, "Insert with wrong embedding size must fail"

    # 13) Case-insensitive category testing
    logger.info("13) Case-insensitive category testing")

    # Clear to start fresh for this test
    db.clear_all_records()

    # Test 13.1: Insert with different case categories - all should normalize to lowercase
    logger.info(" 13.1) Testing category normalization on insert")

    # Use more distinct questions/answers to avoid any deduplication issues
    test_cases = [
        (
            "What is the minimum balance for Banking account?",
            "No minimum balance required",
            "Banking",
        ),
        ("How to open a BANKING account online?", "Visit our website and apply", "BANKING"),
        ("Can I get a banking debit card?", "Yes, free debit card included", "banking"),
        ("Is mobile BaNkInG available?", "Yes, download our mobile app", "BaNkInG"),
    ]

    inserted_count = 0
    for q, a, cat in test_cases:
        ok = db.insert_qa(
            q, a, cat, generate_fake_embedding(q, emb_size), generate_fake_embedding(a, emb_size)
        )
        assert ok, f"Failed to insert {q} with category {cat}"
        if ok:
            inserted_count += 1
            logger.info("  - Inserted: %s", q[:50])

    assert inserted_count == 4, f"Expected 4 insertions, got {inserted_count}"

    # All should be stored as lowercase "banking"
    cats = db.get_categories()
    assert len(cats) == 1, f"Expected 1 unique category, got {len(cats)}: {cats}"
    assert cats[0] == "banking", f"Expected 'banking', got '{cats[0]}'"
    logger.info("  - All case variants stored as 'banking'")

    # Test 13.2: Search with different case categories
    logger.info(" 13.2) Testing case-insensitive category search")
    query_emb = generate_fake_embedding("banking query", emb_size)

    if can_search:
        # Search with different cases should all return same results
        results_lower = db.search_similar_questions(
            query_emb, category="banking", top_k=10, threshold=0.0
        )
        results_upper = db.search_similar_questions(
            query_emb, category="BANKING", top_k=10, threshold=0.0
        )
        results_mixed = db.search_similar_questions(
            query_emb, category="BaNkInG", top_k=10, threshold=0.0
        )

        # Should find 4 results (or less if deduplication occurs)
        # The important thing is all three searches return the same count
        assert len(results_lower) == len(results_upper), (
            f"Different results for 'banking' ({len(results_lower)}) vs 'BANKING' ({len(results_upper)})"
        )
        assert len(results_upper) == len(results_mixed), (
            f"Different results for 'BANKING' ({len(results_upper)}) vs 'BaNkInG' ({len(results_mixed)})"
        )

        # Verify we have at least some results
        assert len(results_lower) > 0, "Should have found at least 1 result"

        logger.info("  - Search with 'banking': %d results", len(results_lower))
        logger.info("  - Search with 'BANKING': %d results", len(results_upper))
        logger.info("  - Search with 'BaNkInG': %d results", len(results_mixed))

        # Verify all results have the normalized category
        for r in results_lower:
            assert r["category"] == "banking", f"Result has wrong category: {r['category']}"

    # Test 13.3: Update category with different cases
    logger.info(" 13.3) Testing category update normalization")

    # Update with uppercase - should normalize to lowercase
    ok = db.update_category("What is the minimum balance for Banking account?", "FINANCE")
    assert ok, "Failed to update category"

    found = db.find_question("What is the minimum balance for Banking account?")
    assert found is not None, "Question not found after update"
    assert found["category"] == "finance", f"Expected 'finance', got '{found['category']}'"

    # Update with mixed case - should normalize to lowercase
    ok = db.update_category("How to open a BANKING account online?", "FiNaNcE")
    assert ok, "Failed to update category"

    found = db.find_question("How to open a BANKING account online?")
    assert found is not None, "Question not found after update"
    assert found["category"] == "finance", f"Expected 'finance', got '{found['category']}'"

    logger.info("  - Categories correctly normalized to lowercase on update")

    # 14) NEW: Test deduplication with different formatting
    logger.info("14) Test deduplication with different question formatting")
    db.clear_all_records()

    # Insert question with specific formatting
    original_q = "Can I use Apple Pay with this card?!"
    ok1 = db.insert_qa(
        original_q,
        "Yes, Apple Pay is supported",
        "Features",
        generate_fake_embedding(original_q, emb_size),
        generate_fake_embedding("Yes", emb_size),
    )
    assert ok1, "Failed to insert original question"

    # Try to insert same question with different formatting - should be rejected
    duplicate_q = "can i use apple pay with this card"
    ok2 = db.insert_qa(
        duplicate_q,
        "Different answer",
        "Different",
        generate_fake_embedding(duplicate_q, emb_size),
        generate_fake_embedding("No", emb_size),
    )
    assert not ok2, "Duplicate with different formatting should be rejected"

    # Verify only one record exists
    all_records = db.get_all_qa_records()
    assert len(all_records) == 1, f"Expected 1 record, got {len(all_records)}"

    # Verify original formatting is preserved
    assert all_records[0]["question"] == original_q.strip(), (
        f"Original formatting not preserved: expected '{original_q.strip()}', "
        f"got '{all_records[0]['question']}'"
    )
    logger.info("  - Deduplication works across different formatting")
    logger.info("  - Original formatting preserved: %s", all_records[0]["question"])

    # Finalize
    logger.info("15) Cleanup and close")
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
