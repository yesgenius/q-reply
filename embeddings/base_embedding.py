# q-reply/embeddings/base_embedding.py
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

from __future__ import annotations

import json
import logging
import time
from typing import Any

from gigachat.client import GigaChatClient
from utils.logger import get_logger


logger = get_logger(__name__)

# Initialize GigaChat client
llm = GigaChatClient()

# Default embedding model
DEFAULT_MODEL = "EmbeddingsGigaR"  # Use instruction-aware model by default

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # Base delay in seconds


def get_instruction(task_type: str | None = None, **kwargs: Any) -> str:
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


def prepare_texts(texts: str | list[str], instruction: str = "", **kwargs: Any) -> list[str]:
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


def _call_api_with_retry(
    llm_client: GigaChatClient,
    prepared_texts: list[str],
    model: str,
    api_params: dict[str, Any],
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> dict[str, Any]:
    """Call GigaChat API with retry logic for empty/invalid responses.

    Args:
        llm_client: GigaChat client instance.
        prepared_texts: List of texts with instructions applied.
        model: Model name for embeddings.
        api_params: Additional API parameters.
        max_retries: Maximum number of retry attempts.
        retry_delay: Base delay between retries (exponential backoff).

    Returns:
        API response dictionary.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            # Call API
            response = llm_client.create_embeddings(
                input_texts=prepared_texts, model=model, **api_params
            )

            # Validate response structure
            if not isinstance(response, dict):
                raise ValueError(f"Invalid response type: {type(response)}")

            if "data" not in response:
                raise ValueError("Missing 'data' field in response")

            # Success
            return response

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # These errors indicate empty/malformed response
            last_error = e

            if attempt < max_retries - 1:
                # Calculate delay with exponential backoff
                delay = retry_delay * (2**attempt)
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                # All retries exhausted
                logger.error(f"All {max_retries} attempts failed. Last error: {e}")

    # Should not reach here, but for type safety
    raise RuntimeError(f"API call failed after {max_retries} attempts: {last_error}")


def create_embeddings(
    texts: str | list[str],
    model: str = DEFAULT_MODEL,
    task_type: str | None = None,
    apply_instruction: bool = True,
    custom_instruction: str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    **kwargs: Any,
) -> tuple[list[list[float]], list[str]]:
    """Create embeddings for input texts with optional instructions.

    This is the main function for creating embeddings. It handles:
    - Single text or batch processing
    - Automatic instruction selection based on task type
    - Custom instruction override
    - Retry logic for empty/invalid API responses
    - Proper error handling and logging

    Args:
        texts: Single text or list of texts to embed.
        model: Model to use ('Embeddings' or 'EmbeddingsGigaR').
        task_type: Type of embedding task for automatic instruction selection.
        apply_instruction: Whether to apply instruction prefix.
        custom_instruction: Override automatic instruction with custom one.
        max_retries: Maximum number of retry attempts for API calls.
        retry_delay: Base delay between retries (exponential backoff).
        **kwargs: Additional parameters passed to instruction generator and API.
            Common keys:
            - categories: For classification tasks
            - x_request_id: Request tracing ID
            - x_session_id: Session tracing ID

    Returns:
        Tuple containing:
            - List of embedding vectors (list of floats for each text)
            - List of actual input texts sent to the model (with instructions)

    Raises:
        ValueError: If inputs are invalid.
        RuntimeError: If embedding creation fails after all retries.

    Examples:
        >>> # Document embedding (no instruction)
        >>> doc_emb, input_texts = create_embeddings("Product description", task_type="document")

        >>> # Query embedding (with instruction)
        >>> query_emb, input_texts = create_embeddings("What is this product?", task_type="query")

        >>> # Batch processing with custom retry settings
        >>> embs, input_texts = create_embeddings(
        ...     ["text1", "text2"], task_type="similarity", max_retries=5, retry_delay=2.0
        ... )
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
        f"text_count={len(texts)}, apply_instruction={apply_instruction}, "
        f"max_retries={max_retries}"
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
    api_params: dict[str, Any] = {}
    tracing_params = ["x_request_id", "x_session_id", "x_client_id"]
    for key in tracing_params:
        if key in kwargs:
            api_params[key] = kwargs.pop(key)

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
                logger.debug(f"  Sample {i + 1}: {preview}")
            if len(prepared_texts) > 2:
                logger.debug("  ...")

            # Show additional API parameters
            if api_params:
                logger.debug("Additional API Parameters:")
                for key, value in api_params.items():
                    logger.debug(f"  {key}: {value}")
            logger.debug("-" * 60)

        # Call API with retry logic
        response = _call_api_with_retry(
            llm, prepared_texts, model, api_params, max_retries, retry_delay
        )

        embeddings_data = response["data"]

        # Validate response count
        if len(embeddings_data) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings_data)}"
            )

        # Extract embedding vectors
        embeddings: list[list[float]] = []
        for i, emb_data in enumerate(embeddings_data):
            if not isinstance(emb_data, dict):
                raise RuntimeError(f"Invalid embedding data at index {i}: {type(emb_data)}")

            if "embedding" not in emb_data:
                raise RuntimeError(f"Missing 'embedding' field for text {i}")

            embedding = emb_data["embedding"]
            if not isinstance(embedding, list):
                raise RuntimeError(f"Invalid embedding type at index {i}: {type(embedding)}")

            embeddings.append(embedding)

        # Log success
        if embeddings:
            logger.info(f"Created {len(embeddings)} embeddings, dimension: {len(embeddings[0])}")

        return embeddings, prepared_texts

    except Exception as e:
        logger.error(f"Failed to create embeddings: {e}")
        raise RuntimeError(f"Embedding creation failed: {e}") from e


def create_batch_embeddings(
    texts: list[str],
    batch_size: int = 100,
    model: str = DEFAULT_MODEL,
    task_type: str | None = None,
    **kwargs: Any,
) -> tuple[list[list[float]], list[str]]:
    """Create embeddings for large text collections in batches.

    Handles large datasets by processing in batches to avoid API limits
    and memory issues. Retry settings are passed through to create_embeddings.

    Args:
        texts: List of texts to embed.
        batch_size: Maximum texts per API call (default: 100).
        model: Embedding model to use.
        task_type: Type of embedding task.
        **kwargs: Additional parameters passed to create_embeddings,
            including max_retries and retry_delay.

    Returns:
        Tuple containing:
            - List of embedding vectors for all input texts
            - List of all actual input texts sent to the model (with instructions)

    Raises:
        ValueError: If texts is empty or batch_size is invalid.
        RuntimeError: If batch processing fails.

    Examples:
        >>> texts = ["text1", "text2", ..., "text1000"]
        >>> embeddings, input_texts = create_batch_embeddings(texts, batch_size=50, max_retries=5)
    """
    if not texts:
        return [], []

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    logger.info(f"Processing {len(texts)} texts in batches of {batch_size}")

    all_embeddings: list[list[float]] = []
    all_input_texts: list[str] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.debug(f"Processing batch {batch_num}/{total_batches}, size: {len(batch)}")

        try:
            batch_embeddings, batch_input_texts = create_embeddings(
                texts=batch, model=model, task_type=task_type, **kwargs
            )
            all_embeddings.extend(batch_embeddings)
            all_input_texts.extend(batch_input_texts)

        except Exception as e:
            logger.error(f"Failed to process batch {batch_num} starting at index {i}: {e}")
            raise RuntimeError(f"Batch processing failed at batch {batch_num}") from e

    logger.info(f"Successfully created {len(all_embeddings)} embeddings")
    return all_embeddings, all_input_texts


# Test section
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s][%(filename)s:%(lineno)d][%(message)s]",
    )

    from collections.abc import Callable
    import sys

    print("=== Base Embedding Module Tests ===\n")

    # Test configuration
    tests_passed = 0
    tests_failed = 0
    tests_skipped = 0

    def run_test(
        test_name: str, test_func: Callable[[], None], skip_on_api_error: bool = False
    ) -> None:
        """Run a single test with proper error handling."""
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

    # Test 1: Single text embedding with input_texts return
    def test_single_text() -> None:
        """Test embedding creation for single text with input_texts return."""
        text = "Тестовый текст для проверки"
        result, input_texts = create_embeddings(text, task_type="document")

        # Check embeddings
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) == 1, f"Expected 1 embedding, got {len(result)}"
        assert isinstance(result[0], list), f"Expected list of floats, got {type(result[0])}"
        assert len(result[0]) > 0, "Embedding vector is empty"
        assert all(isinstance(x, (int, float)) for x in result[0]), (
            "Embedding should contain numbers"
        )

        # Check input_texts
        assert isinstance(input_texts, list), f"Expected list, got {type(input_texts)}"
        assert len(input_texts) == 1, f"Expected 1 input text, got {len(input_texts)}"
        assert input_texts[0] == text, "Input text should match original for document type"

    run_test("Single text embedding", test_single_text, skip_on_api_error=True)

    # Test 2: Multiple texts batch with input_texts
    def test_multiple_texts() -> None:
        """Test batch processing of multiple texts with input_texts return."""
        texts = ["Текст 1", "Текст 2", "Текст 3"]
        results, input_texts = create_embeddings(texts, task_type="document")

        # Check embeddings
        assert len(results) == 3, f"Expected 3 embeddings, got {len(results)}"

        # Check all have same dimension
        first_dim = len(results[0])
        assert first_dim > 0, "Embedding dimension is 0"

        for i, emb in enumerate(results):
            assert len(emb) == first_dim, (
                f"Embedding {i} has different dimension: {len(emb)} vs {first_dim}"
            )
            assert isinstance(emb, list), f"Embedding {i} is not a list"

        # Check input_texts
        assert len(input_texts) == 3, f"Expected 3 input texts, got {len(input_texts)}"
        assert input_texts == texts, "Input texts should match originals for document type"

    run_test("Multiple texts batch", test_multiple_texts, skip_on_api_error=True)

    # Test 3: Query with instruction - verify input_texts
    def test_query_with_instruction() -> None:
        """Test that query task type adds instruction to input_texts."""
        text = "Что такое Python?"
        _, input_texts = create_embeddings(text, task_type="query")

        expected_prefix = "Дан вопрос, найди семантически похожие вопросы\nвопрос: "
        assert len(input_texts) == 1, "Should have one input text"
        assert input_texts[0].startswith(expected_prefix), (
            f"Input text should start with instruction.\n"
            f"Expected prefix: {expected_prefix}\n"
            f"Got: {input_texts[0][: len(expected_prefix)]}"
        )
        assert text in input_texts[0], "Original text should be in input_texts"

    run_test("Query with instruction", test_query_with_instruction, skip_on_api_error=True)

    # Test 4: Custom retry settings
    def test_custom_retry_settings() -> None:
        """Test custom retry settings are passed through."""
        text = "Test text for retry"

        # This should work with custom settings
        _, _ = create_embeddings(text, task_type="document", max_retries=2, retry_delay=0.5)
        # If it doesn't throw, the settings were accepted

    run_test("Custom retry settings", test_custom_retry_settings, skip_on_api_error=True)

    # Test 5: Batch embeddings with retry
    def test_batch_with_retry() -> None:
        """Test batch processing with retry settings."""
        texts = [f"Text {i}" for i in range(7)]

        batch_embs, batch_input_texts = create_batch_embeddings(
            texts, batch_size=3, task_type="document", max_retries=2, retry_delay=1.0
        )

        # Check embeddings
        assert len(batch_embs) == 7, f"Expected 7 embeddings, got {len(batch_embs)}"

        # Check all have same dimension
        first_dim = len(batch_embs[0])
        for i, emb in enumerate(batch_embs):
            assert len(emb) == first_dim, f"Embedding {i} has different dimension"

        # Check input_texts
        assert len(batch_input_texts) == 7, f"Expected 7 input texts, got {len(batch_input_texts)}"

    run_test("Batch with retry", test_batch_with_retry, skip_on_api_error=True)

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
