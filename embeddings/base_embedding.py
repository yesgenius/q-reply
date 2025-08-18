"""Base embedding template for semantic search and similarity tasks.

This module provides a universal template for creating embedding modules
with consistent structure, error handling, and instruction support.

The module supports instruction-based embeddings for asymmetric tasks
(e.g., query-document retrieval) and symmetric tasks (e.g., semantic similarity).

Example:
    To create a new embedding module, copy this file and modify:
    1. Update module docstring with specific use case
    2. Change model if needed (Embeddings vs EmbeddingsGigaR)
    3. Modify get_instruction() with your specific instruction template
    4. Update test section with relevant examples

Usage:
    python -m embeddings.base_embedding
"""

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

from gigachat.client import GigaChatClient

# Initialize GigaChat client
llm = GigaChatClient()

# Default embedding model
DEFAULT_MODEL = "EmbeddingsGigaR"  # Use instruction-aware model by default


def get_instruction(task_type: Optional[str] = None, **kwargs: Any) -> str:
    """Generate instruction prefix for embedding tasks.

    This method should be overridden in specific embedding implementations
    to provide task-specific instructions for better retrieval quality.

    Args:
        task_type: Type of embedding task. Common values:
            - None or "document": No instruction (for indexing documents)
            - "query": For retrieval queries
            - "classification": For classification tasks
            - "similarity": For symmetric similarity tasks
            - "paraphrase": For paraphrase detection
            - "query_answer": For finding answers by question
        **kwargs: Additional parameters for instruction generation.
            Common keys might include:
            - categories: List of categories for classification
            - language: Language of the content

    Returns:
        Instruction string to prepend to the text, or empty string if no instruction.

    Examples:
        >>> get_instruction("query")
        'Дан вопрос, найди семантически похожие вопросы\\nвопрос: '
        >>> get_instruction("document")
        ''
    """
    # No instruction for documents (asymmetric retrieval best practice)
    if task_type is None or task_type == "document":
        return ""

    # Example instruction patterns for common tasks
    if task_type == "query":
        # Default retrieval instruction for question search
        return "Дан вопрос, найди семантически похожие вопросы\nвопрос: "

    if task_type == "query_answer":
        # Instruction for finding answers by question
        return "Дан вопрос, необходимо найти абзац текста с ответом\nвопрос: "

    if task_type == "similarity":
        # Symmetric similarity instruction
        return "Дан текст, найди семантически похожие тексты\nтекст: "

    if task_type == "paraphrase":
        # Paraphrase detection instruction
        return "Дан текст, необходимо найти его парафраз\nтекст: "

    if task_type == "classification" and "categories" in kwargs:
        # Classification with categories
        categories = kwargs["categories"]
        if not isinstance(categories, list):
            logger.warning(f"Categories should be a list, got {type(categories)}")
            return ""
        categories_str = ", ".join(str(cat) for cat in categories)
        return f"Классифицируй вопрос по одной из категорий из списка: {categories_str}\nвопрос: "

    # Return empty string for unknown task types
    logger.warning(f"Unknown task_type: {task_type}. Using no instruction.")
    return ""


def prepare_texts(
    texts: Union[str, List[str]], instruction: str = "", **kwargs: Any
) -> List[str]:
    """Prepare texts for embedding with optional instruction prefix.

    Args:
        texts: Single text or list of texts to prepare.
        instruction: Instruction to prepend to each text.
        **kwargs: Additional parameters (reserved for future use).

    Returns:
        List of prepared texts ready for embedding.

    Raises:
        ValueError: If input texts are empty.

    Examples:
        >>> prepare_texts("Hello world", "Classify text: ")
        ['Classify text: Hello world']
        >>> prepare_texts(["text1", "text2"], "")
        ['text1', 'text2']
    """
    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]

    # Validate input
    if not texts:
        raise ValueError("Input texts cannot be empty")

    # Apply instruction if provided
    if instruction:
        prepared = [f"{instruction}{text}" for text in texts]
        logger.debug(f"Applied instruction to {len(prepared)} texts")
    else:
        prepared = texts[:]  # Create a copy to avoid modifying original
        logger.debug(f"No instruction applied to {len(prepared)} texts")

    return prepared


def create_embeddings(
    texts: Union[str, List[str]],
    model: str = DEFAULT_MODEL,
    task_type: Optional[str] = None,
    apply_instruction: bool = True,
    custom_instruction: Optional[str] = None,
    **kwargs: Any,
) -> List[List[float]]:
    """Create embeddings for input texts with optional instructions.

    This is the main function for creating embeddings. It handles:
    - Single text or batch processing
    - Automatic instruction selection based on task type
    - Custom instruction override
    - Proper error handling and logging

    Args:
        texts: Single text or list of texts to embed.
        model: Model to use ('Embeddings' or 'EmbeddingsGigaR').
        task_type: Type of embedding task for automatic instruction selection.
        apply_instruction: Whether to apply instruction prefix.
        custom_instruction: Override automatic instruction with custom one.
        **kwargs: Additional parameters passed to instruction generator and API.
            Common keys:
            - categories: For classification tasks
            - x_request_id: Request tracing ID
            - x_session_id: Session tracing ID

    Returns:
        List of embedding vectors (list of floats for each text).

    Raises:
        ValueError: If inputs are invalid.
        RuntimeError: If embedding creation fails.

    Examples:
        >>> # Document embedding (no instruction)
        >>> doc_emb = create_embeddings("Product description", task_type="document")

        >>> # Query embedding (with instruction)
        >>> query_emb = create_embeddings("What is this product?", task_type="query")

        >>> # Batch processing
        >>> embs = create_embeddings(["text1", "text2"], task_type="similarity")
    """
    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]

    # Validate input
    if not texts:
        raise ValueError("Input texts cannot be empty")

    # Log request details
    logger.debug(
        f"Creating embeddings: model={model}, task_type={task_type}, "
        f"text_count={len(texts)}, apply_instruction={apply_instruction}"
    )

    # Determine instruction to use
    instruction = ""
    if apply_instruction:
        if custom_instruction is not None:
            instruction = custom_instruction
            logger.debug(f"Using custom instruction: {instruction[:50]}...")
        else:
            instruction = get_instruction(task_type, **kwargs)
            if instruction:
                logger.debug(f"Using task instruction: {instruction[:50]}...")

    # Prepare texts with instruction
    prepared_texts = prepare_texts(texts, instruction)

    # Extract API tracing parameters from kwargs
    # These are optional headers for request tracing/debugging
    api_params: Dict[str, Any] = {}
    tracing_params = ["x_request_id", "x_session_id", "x_client_id"]
    for key in tracing_params:
        if key in kwargs:
            api_params[key] = kwargs.pop(key)  # Remove from kwargs to avoid duplication

    try:
        # Log API call parameters (debug mode only)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("-" * 60)
            logger.debug("Calling GigaChat API with the following parameters:")
            logger.debug(f"Model: {model}")
            logger.debug(f"Input Texts: list with {len(prepared_texts)} items")

            # Show sample texts
            for i, text in enumerate(prepared_texts[:2]):
                preview = text[:200] + "..." if len(text) > 200 else text
                logger.debug(f"  Sample {i+1}: {preview}")
            if len(prepared_texts) > 2:
                logger.debug("  ...")

            # Show additional API parameters
            if api_params:
                logger.debug("Additional API Parameters:")
                for key, value in api_params.items():
                    logger.debug(f"  {key}: {value}")
            logger.debug("-" * 60)

        # Call GigaChat API
        # Note: model parameter is passed directly, not through kwargs
        response = llm.create_embeddings(
            input_texts=prepared_texts, model=model, **api_params
        )

        # Validate response structure
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected response type: {type(response)}")

        if "data" not in response:
            raise RuntimeError(
                f"Missing 'data' field in response: {list(response.keys())}"
            )

        embeddings_data = response["data"]

        # Validate response count
        if len(embeddings_data) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings_data)}"
            )

        # Extract embedding vectors
        embeddings: List[List[float]] = []
        for i, emb_data in enumerate(embeddings_data):
            if not isinstance(emb_data, dict):
                raise RuntimeError(
                    f"Invalid embedding data at index {i}: {type(emb_data)}"
                )

            if "embedding" not in emb_data:
                raise RuntimeError(f"Missing 'embedding' field for text {i}")

            embedding = emb_data["embedding"]
            if not isinstance(embedding, list):
                raise RuntimeError(
                    f"Invalid embedding type at index {i}: {type(embedding)}"
                )

            embeddings.append(embedding)

        # Log success
        if embeddings:
            logger.info(
                f"Created {len(embeddings)} embeddings, dimension: {len(embeddings[0])}"
            )

        return embeddings

    except Exception as e:
        logger.error(f"Failed to create embeddings: {e}")
        raise RuntimeError(f"Embedding creation failed: {e}") from e


def create_batch_embeddings(
    texts: List[str],
    batch_size: int = 100,
    model: str = DEFAULT_MODEL,
    task_type: Optional[str] = None,
    **kwargs: Any,
) -> List[List[float]]:
    """Create embeddings for large text collections in batches.

    Handles large datasets by processing in batches to avoid API limits
    and memory issues.

    Args:
        texts: List of texts to embed.
        batch_size: Maximum texts per API call (default: 100).
        model: Embedding model to use.
        task_type: Type of embedding task.
        **kwargs: Additional parameters passed to create_embeddings.

    Returns:
        List of embedding vectors for all input texts.

    Raises:
        []: If texts is empty or batch_size is invalid.
        RuntimeError: If batch processing fails.

    Examples:
        >>> texts = ["text1", "text2", ..., "text1000"]
        >>> embeddings = create_batch_embeddings(texts, batch_size=50)
    """
    if not texts:
        return []

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    logger.info(f"Processing {len(texts)} texts in batches of {batch_size}")

    all_embeddings: List[List[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.debug(
            f"Processing batch {batch_num}/{total_batches}, size: {len(batch)}"
        )

        try:
            batch_embeddings = create_embeddings(
                texts=batch, model=model, task_type=task_type, **kwargs
            )
            all_embeddings.extend(batch_embeddings)

        except Exception as e:
            logger.error(
                f"Failed to process batch {batch_num} starting at index {i}: {e}"
            )
            raise RuntimeError(f"Batch processing failed at batch {batch_num}") from e

    logger.info(f"Successfully created {len(all_embeddings)} embeddings")
    return all_embeddings


# Test section
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # DEBUG
        format="[%(asctime)s][%(name)s][%(levelname)s][%(filename)s:%(lineno)d][%(message)s]",
    )

    import sys
    from typing import Callable

    print("=== Base Embedding Module Tests ===\n")

    # Test configuration
    tests_passed = 0
    tests_failed = 0
    tests_skipped = 0

    def run_test(
        test_name: str, test_func: Callable[[], None], skip_on_api_error: bool = False
    ) -> None:
        """Run a single test with proper error handling.

        Args:
            test_name: Name of the test for reporting.
            test_func: Test function to execute.
            skip_on_api_error: Skip test if API returns 402 (payment required).
        """
        global tests_passed, tests_failed, tests_skipped

        try:
            print(f"Running: {test_name}")
            test_func()
            print(f"✅ PASSED: {test_name}\n")
            tests_passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {test_name}")
            print(f"   Assertion error: {e}\n")
            tests_failed += 1
        except Exception as e:
            # Check if it's a payment required error
            if skip_on_api_error and "402" in str(e):
                print(f"⚠️  SKIPPED: {test_name} (API requires payment)\n")
                tests_skipped += 1
            else:
                print(f"❌ ERROR: {test_name}")
                print(f"   Exception: {e}\n")
                tests_failed += 1

    # Test 1: Single text embedding
    def test_single_text() -> None:
        """Test embedding creation for single text."""
        text = "Тестовый текст для проверки"
        result = create_embeddings(text, task_type="document")

        # Assertions
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) == 1, f"Expected 1 embedding, got {len(result)}"
        assert isinstance(
            result[0], list
        ), f"Expected list of floats, got {type(result[0])}"
        assert len(result[0]) > 0, "Embedding vector is empty"
        assert all(
            isinstance(x, (int, float)) for x in result[0]
        ), "Embedding should contain numbers"

    run_test("Single text embedding", test_single_text, skip_on_api_error=True)

    # Test 2: Multiple texts batch
    def test_multiple_texts() -> None:
        """Test batch processing of multiple texts."""
        texts = ["Текст 1", "Текст 2", "Текст 3"]
        results = create_embeddings(texts, task_type="document")

        # Assertions
        assert len(results) == 3, f"Expected 3 embeddings, got {len(results)}"

        # Check all have same dimension
        first_dim = len(results[0])
        assert first_dim > 0, "Embedding dimension is 0"

        for i, emb in enumerate(results):
            assert (
                len(emb) == first_dim
            ), f"Embedding {i} has different dimension: {len(emb)} vs {first_dim}"
            assert isinstance(emb, list), f"Embedding {i} is not a list"

    run_test("Multiple texts batch", test_multiple_texts, skip_on_api_error=True)

    # Test 3: Task type "document" (no instruction)
    def test_task_type_document() -> None:
        """Test document task type (should have no instruction)."""
        text = "Документ для индексации"

        # Test that document type doesn't add instruction
        doc_emb = create_embeddings(text, task_type="document")
        no_instruction_emb = create_embeddings(text, apply_instruction=False)

        assert isinstance(doc_emb, list), "Should return list"
        assert len(doc_emb) == 1, "Should return one embedding"

        # Note: We can't directly compare embeddings due to API variations,
        # but we check they have same structure
        assert len(doc_emb[0]) == len(no_instruction_emb[0]), "Dimensions should match"

    run_test("Task type: document", test_task_type_document, skip_on_api_error=True)

    # Test 4: Task type "query"
    def test_task_type_query() -> None:
        """Test query task type with instruction."""
        text = "Что такое Python?"

        # Create embeddings with different task types
        doc_emb = create_embeddings(text, task_type="document")
        query_emb = create_embeddings(text, task_type="query")

        assert len(query_emb) == 1, "Should return one embedding"
        assert len(query_emb[0]) == len(doc_emb[0]), "Dimensions should be same"

        # Embeddings should be different due to instruction
        # We check at least some values differ
        differences = sum(
            1 for a, b in zip(doc_emb[0][:10], query_emb[0][:10]) if abs(a - b) > 1e-6
        )
        assert differences > 0, "Query and document embeddings should differ"

    run_test("Task type: query", test_task_type_query, skip_on_api_error=True)

    # Test 5: Task type "query_answer"
    def test_task_type_query_answer() -> None:
        """Test query_answer task type."""
        text = "Как работает нейронная сеть?"

        query_answer_emb = create_embeddings(text, task_type="query_answer")

        assert isinstance(query_answer_emb, list), "Should return list"
        assert len(query_answer_emb) == 1, "Should return one embedding"
        assert len(query_answer_emb[0]) > 0, "Embedding should not be empty"

    run_test(
        "Task type: query_answer", test_task_type_query_answer, skip_on_api_error=True
    )

    # Test 6: Task type "similarity"
    def test_task_type_similarity() -> None:
        """Test similarity task type for symmetric search."""
        texts = ["Первый текст", "Второй текст"]

        # Both texts should get same instruction
        embs = create_embeddings(texts, task_type="similarity")

        assert len(embs) == 2, f"Expected 2 embeddings, got {len(embs)}"
        assert len(embs[0]) == len(embs[1]), "Embeddings should have same dimension"

        # Check embeddings are different (not comparing floats directly)
        differences = sum(1 for a, b in zip(embs[0], embs[1]) if abs(a - b) > 1e-6)
        assert differences > 0, "Different texts should have different embeddings"

    run_test("Task type: similarity", test_task_type_similarity, skip_on_api_error=True)

    # Test 7: Task type "paraphrase"
    def test_task_type_paraphrase() -> None:
        """Test paraphrase detection task type."""
        original = "Как установить Python?"
        paraphrase = "Как инсталлировать питон?"

        orig_emb = create_embeddings(original, task_type="paraphrase")
        para_emb = create_embeddings(paraphrase, task_type="paraphrase")

        assert len(orig_emb) == 1, "Should return one embedding"
        assert len(para_emb) == 1, "Should return one embedding"
        assert len(orig_emb[0]) == len(para_emb[0]), "Same dimension expected"

        # Check embeddings are different
        differences = sum(
            1 for a, b in zip(orig_emb[0], para_emb[0]) if abs(a - b) > 1e-6
        )
        assert differences > 0, "Different texts should have different embeddings"

    run_test("Task type: paraphrase", test_task_type_paraphrase, skip_on_api_error=True)

    # Test 8: Task type "classification" with categories
    def test_task_type_classification() -> None:
        """Test classification task type with categories."""
        text = "Как вернуть товар?"
        categories = ["техподдержка", "продажи", "возврат"]

        class_emb = create_embeddings(
            text, task_type="classification", categories=categories
        )

        assert isinstance(class_emb, list), "Should return list"
        assert len(class_emb) == 1, "Should return one embedding"
        assert len(class_emb[0]) > 0, "Embedding should not be empty"

    run_test(
        "Task type: classification",
        test_task_type_classification,
        skip_on_api_error=True,
    )

    # Test 9: Custom instruction override
    def test_custom_instruction() -> None:
        """Test custom instruction override."""
        text = "Test text"
        custom_instruction = "Custom instruction for testing\ntext: "

        emb = create_embeddings(text, custom_instruction=custom_instruction)

        assert isinstance(emb, list), "Should return list"
        assert len(emb) == 1, "Should return one embedding"
        assert len(emb[0]) > 0, "Embedding should not be empty"

    run_test("Custom instruction", test_custom_instruction, skip_on_api_error=True)

    # Test 10: Unknown task type handling
    def test_unknown_task_type() -> None:
        """Test handling of unknown task type."""
        text = "Test text"

        # Should log warning but still work (returns empty instruction)
        emb = create_embeddings(text, task_type="unknown_type")

        assert isinstance(emb, list), "Should still return list"
        assert len(emb) == 1, "Should return one embedding"

    run_test("Unknown task type", test_unknown_task_type, skip_on_api_error=True)

    # Test 11: Empty input validation
    def test_empty_input() -> None:
        """Test that empty input raises ValueError."""
        try:
            create_embeddings([])
            assert False, "Should have raised ValueError for empty input"
        except ValueError as e:
            assert (
                "empty" in str(e).lower()
            ), f"Error message should mention 'empty': {e}"

    run_test("Empty input validation", test_empty_input)

    # Test 12: Apply instruction flag
    def test_apply_instruction_flag() -> None:
        """Test apply_instruction parameter."""
        text = "Test text"

        # With instruction disabled
        no_inst = create_embeddings(
            text,
            task_type="query",  # This would normally add instruction
            apply_instruction=False,  # But we disable it
        )

        # Same as document type
        doc = create_embeddings(text, task_type="document")

        assert len(no_inst) == 1, "Should return one embedding"
        assert len(doc) == 1, "Should return one embedding"
        assert len(no_inst[0]) == len(doc[0]), "Should have same dimension"

    run_test(
        "Apply instruction flag", test_apply_instruction_flag, skip_on_api_error=True
    )

    # Test 13: Batch embeddings function
    def test_batch_embeddings() -> None:
        """Test create_batch_embeddings function."""
        # Create test data
        texts = [f"Text {i}" for i in range(7)]

        batch_embs = create_batch_embeddings(
            texts, batch_size=3, task_type="document"  # Process in batches of 3
        )

        assert len(batch_embs) == 7, f"Expected 7 embeddings, got {len(batch_embs)}"

        # Check all have same dimension
        first_dim = len(batch_embs[0])
        for i, emb in enumerate(batch_embs):
            assert len(emb) == first_dim, f"Embedding {i} has different dimension"

    run_test("Batch embeddings", test_batch_embeddings, skip_on_api_error=True)

    # Test 14: Instruction consistency
    def test_instruction_consistency() -> None:
        """Test that same task type always gives same instruction."""
        instruction1 = get_instruction("query")
        instruction2 = get_instruction("query")

        assert (
            instruction1 == instruction2
        ), "Same task type should give same instruction"
        assert (
            instruction1 == "Дан вопрос, найди семантически похожие вопросы\nвопрос: "
        ), f"Unexpected instruction: {instruction1}"

    run_test("Instruction consistency", test_instruction_consistency)

    # Test 15: Prepare texts function
    def test_prepare_texts() -> None:
        """Test prepare_texts helper function."""
        # Single text
        result = prepare_texts("test", "prefix: ")
        assert result == ["prefix: test"], f"Unexpected result: {result}"

        # Multiple texts
        result = prepare_texts(["t1", "t2"], "prefix: ")
        assert result == ["prefix: t1", "prefix: t2"], f"Unexpected result: {result}"

        # No instruction
        result = prepare_texts("test", "")
        assert result == ["test"], f"Unexpected result: {result}"

    run_test("Prepare texts helper", test_prepare_texts)

    # Print test summary
    print("=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Passed:  {tests_passed}")
    print(f"❌ Failed:  {tests_failed}")
    print(f"⚠️  Skipped: {tests_skipped}")
    print("=" * 50)

    # Exit with appropriate code
    if tests_failed > 0:
        print("\n❌ TESTS FAILED!")
        sys.exit(1)
    else:
        print("\n✅ ALL TESTS PASSED!")
        sys.exit(0)
