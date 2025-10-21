# q-reply/prompts/get_answer_prompt_test.py
"""Test module for get_answer_prompt with proper imports.

This test module validates the answer generation prompt functionality,
including context mode switching, message generation, and error handling.
Saves all generated messages to JSON files for analysis.

Example:
    Run all tests:
```bash
    python prompts/get_answer_prompt_test.py
```
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any


# Add parent directory to sys.path to import gigachat
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Now we can import gigachat
try:
    from gigachat.client import GigaChatClient

    print(f"Successfully imported GigaChatClient from: {parent_dir}/gigachat/")
except ImportError as e:
    print(f"Failed to import GigaChatClient: {e}")
    print(f"  Searched in: {parent_dir}/gigachat/")
    print(f"  Current sys.path: {sys.path[:3]}...")
    sys.exit(1)

# Import the module to test (it's in the same directory)
import get_answer_prompt


def save_messages(messages: list[dict[str, str]], filename: str, test_dir: Path) -> None:
    """Save messages to formatted text file with real newlines.

    Args:
        messages: List of message dictionaries to save.
        filename: Name for the output file (without extension).
        test_dir: Directory to save files in.
    """
    txt_path = test_dir / f"{filename}.txt"

    # Custom JSON serialization with real newlines
    def format_json_with_newlines(obj: Any, indent_level: int = 0) -> str:
        indent = "  " * indent_level

        if isinstance(obj, dict):
            if not obj:
                return "{}"
            items = []
            for key, value in obj.items():
                formatted_value = format_json_with_newlines(value, indent_level + 1)
                items.append(f'{indent}  "{key}": {formatted_value}')
            return "{\n" + ",\n".join(items) + f"\n{indent}}}"

        if isinstance(obj, list):
            if not obj:
                return "[]"
            items = []
            for item in obj:
                formatted_item = format_json_with_newlines(item, indent_level + 1)
                items.append(f"{indent}  {formatted_item}")
            return "[\n" + ",\n".join(items) + f"\n{indent}]"

        if isinstance(obj, str):
            # Replace \n with actual newlines while keeping valid JSON string syntax
            escaped = obj.replace("\\", "\\\\").replace('"', '\\"')
            # Now replace literal \n sequences with actual newlines
            escaped = escaped.replace("\\\\n", "\n")
            return f'"{escaped}"'

        return json.dumps(obj, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        formatted = format_json_with_newlines(messages)
        f.write(formatted)

    print(f"    Saved: {txt_path.name}")


def run_tests() -> None:
    """Run all tests with message saving."""
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )

    print("\n" + "=" * 60)
    print("TEST MODULE FOR get_answer_prompt.py")
    print("=" * 60)
    print(f"Working directory: {Path.cwd()}")
    print(f"Test module path: {Path(__file__)}")
    print("=" * 60 + "\n")

    # Create test output directory with module name
    test_script_dir = Path(__file__).parent
    test_module_name = Path(__file__).stem
    test_dir = test_script_dir / test_module_name
    test_dir.mkdir(exist_ok=True)
    print(f"Output directory: {test_dir.absolute()}\n")

    # Track results
    tests_passed = 0
    tests_failed = 0
    failed_tests = []

    # Test 1: Basic initialization
    print("TEST 1: Basic initialization (no topic)")
    print("-" * 40)
    try:
        prompt = get_answer_prompt.update_system_prompt()
        assert prompt is not None, "Prompt is None"
        assert len(prompt) > 0, "Prompt is empty"
        assert isinstance(prompt, str), f"Prompt is {type(prompt)}, not str"

        # Check for new field in prompt
        assert "sources_used_reasoning" in prompt, (
            "New field 'sources_used_reasoning' not in prompt"
        )

        # Save the system prompt
        init_msg = [{"role": "system", "content": prompt}]
        save_messages(init_msg, "msgs_01_init_basic", test_dir)

        print(f"Initialized: {len(prompt):,} characters")
        tests_passed += 1
    except AssertionError as e:
        print(f"Failed: {e}")
        tests_failed += 1
        failed_tests.append("Test 1")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        tests_failed += 1
        failed_tests.append("Test 1")
    print()

    # Test 2: Topic-specific initialization
    print("TEST 2: Topic-specific initialization")
    print("-" * 40)
    try:
        topic = "Cloud Native Architecture and Microservices"
        prompt = get_answer_prompt.update_system_prompt(topic=topic)

        assert prompt is not None, "Prompt is None"
        assert topic in prompt, f"Topic '{topic}' not in prompt"
        assert get_answer_prompt._current_topic == topic, "Topic not cached"

        # Save topic-enhanced prompt
        topic_msg = [{"role": "system", "content": prompt}]
        save_messages(topic_msg, "msgs_02_init_topic", test_dir)

        print(f"Topic set: {topic}")
        tests_passed += 1
    except AssertionError as e:
        print(f"Failed: {e}")
        tests_failed += 1
        failed_tests.append("Test 2")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        tests_failed += 1
        failed_tests.append("Test 2")
    print()

    # Test 3: Knowledge base
    print("TEST 3: Knowledge base integration")
    print("-" * 40)
    try:
        kb_file = Path(get_answer_prompt.__file__).parent / (
            Path(get_answer_prompt.__file__).stem + "_kbase.txt"
        )
        kb_content = """CONFERENCE GUIDELINES:
- Maximum presentation time: 20 minutes
- Q&A session: 5 minutes per question
- All content must be original work"""

        try:
            # Create knowledge base file
            kb_file.write_text(kb_content, encoding="utf-8")

            # Reset and reload
            get_answer_prompt._system_prompt = None
            get_answer_prompt._loaded_knowledge_base = None

            prompt = get_answer_prompt.update_system_prompt(topic="AI Conference")

            assert get_answer_prompt._loaded_knowledge_base is not None, "KB not loaded"
            assert "CONFERENCE GUIDELINES" in prompt, "KB content missing"

            # Save KB-enhanced prompt
            kb_msg = [{"role": "system", "content": prompt}]
            save_messages(kb_msg, "msgs_03_knowledge_base", test_dir)

            print(f"KB loaded: {kb_file.name}")
            tests_passed += 1

        finally:
            # Cleanup
            if kb_file.exists():
                kb_file.unlink()
                print(f"  Cleaned up: {kb_file.name}")

    except AssertionError as e:
        print(f"Failed: {e}")
        tests_failed += 1
        failed_tests.append("Test 3")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        tests_failed += 1
        failed_tests.append("Test 3")
    print()

    # Test 4: Context mode switching (replaces threshold testing)
    print("TEST 4: Context mode switching")
    print("-" * 40)

    # Reset for clean state
    get_answer_prompt._system_prompt = None
    get_answer_prompt._use_chat_history = False
    get_answer_prompt.update_system_prompt(topic="Machine Learning")

    # Pre-filtered QA pairs (no filtering happens in get_answer_prompt anymore)
    qa_pairs: list[get_answer_prompt.QAPair] = [
        {
            "question": "What is transfer learning?",
            "answer": "Reusing pre-trained models for new tasks.",
            "similarity": 0.92,
        },
        {
            "question": "How to prevent overfitting?",
            "answer": "Use regularization, dropout, validation.",
            "similarity": 0.85,
        },
    ]

    test_4_passed = True

    # Sub-test 4a: Chat history mode
    try:
        print("  4a) Chat history mode (use_history=True)")
        get_answer_prompt.update_chat_history(True)

        messages = get_answer_prompt.get_messages("How to improve models?", qa_pairs)

        # Should have: 1 system + 4 history (2 pairs * 2 messages) + 1 user
        expected = 1 + 4 + 1
        assert len(messages) == expected, f"Expected {expected}, got {len(messages)}"

        # Verify history messages contain JSON responses with new field
        for i in range(1, len(messages) - 1):  # Skip system and final user
            if messages[i]["role"] == "assistant":
                content = messages[i]["content"]
                assert "{" in content and "}" in content, (
                    f"Assistant message {i} should contain JSON"
                )
                assert '"answer"' in content, f"Assistant message {i} should have 'answer' field"
                assert '"confidence"' in content, (
                    f"Assistant message {i} should have 'confidence' field"
                )
                assert '"sources_used_reasoning"' in content, (
                    f"Assistant message {i} should have 'sources_used_reasoning' field"
                )

        save_messages(messages, "msgs_04a_chat_history_mode", test_dir)
        print(f"      {len(messages)} messages generated (with history)")
    except AssertionError as e:
        print(f"      Failed: {e}")
        test_4_passed = False

    # Sub-test 4b: User context mode
    try:
        print("  4b) User context mode (use_history=False)")
        get_answer_prompt.update_chat_history(False)

        messages = get_answer_prompt.get_messages("How to improve models?", qa_pairs)

        # Should have only system + user
        assert len(messages) == 2, f"Expected 2 messages (system+user), got {len(messages)}"
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        # Verify context is in user message
        user_content = messages[1]["content"]
        assert "CONTEXT Q&A PAIRS" in user_content, "Context should be in user message"
        # Both pairs should be present in context
        assert "transfer learning" in user_content
        assert "overfitting" in user_content

        save_messages(messages, "msgs_04b_user_context_mode", test_dir)
        print("      All pairs in user context (no history)")
    except AssertionError as e:
        print(f"      Failed: {e}")
        test_4_passed = False

    # Sub-test 4c: Empty QA pairs with chat history mode
    try:
        print("  4c) Empty QA pairs with chat history mode")
        get_answer_prompt.update_chat_history(True)

        messages = get_answer_prompt.get_messages("Test question?", [])

        # Should have only system + user (no history when empty)
        assert len(messages) == 2, f"Expected 2, got {len(messages)}"
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        save_messages(messages, "msgs_04c_empty_qa_history_mode", test_dir)
        print("      No history when empty QA pairs")
    except AssertionError as e:
        print(f"      Failed: {e}")
        test_4_passed = False

    # Sub-test 4d: Single QA pair switching
    try:
        print("  4d) Single QA pair mode switching")
        single_qa: list[get_answer_prompt.QAPair] = [
            {"question": "What is Docker?", "answer": "Container platform.", "similarity": 0.9}
        ]

        # Test with history mode
        get_answer_prompt.update_chat_history(True)
        messages_hist = get_answer_prompt.get_messages("Question?", single_qa)
        expected_hist = 1 + 2 + 1  # system + 1 pair history + user
        assert len(messages_hist) == expected_hist, (
            f"History mode: expected {expected_hist}, got {len(messages_hist)}"
        )

        # Test with context mode
        get_answer_prompt.update_chat_history(False)
        messages_ctx = get_answer_prompt.get_messages("Question?", single_qa)
        expected_ctx = 2  # system + user
        assert len(messages_ctx) == expected_ctx, (
            f"Context mode: expected {expected_ctx}, got {len(messages_ctx)}"
        )

        save_messages(messages_hist, "msgs_04d_single_qa_history", test_dir)
        save_messages(messages_ctx, "msgs_04d_single_qa_context", test_dir)
        print("      Mode switching works correctly")
    except AssertionError as e:
        print(f"      Failed: {e}")
        test_4_passed = False

    if test_4_passed:
        print("Test 4 passed")
        tests_passed += 1
    else:
        print("Test 4 failed")
        tests_failed += 1
        failed_tests.append("Test 4")
    print()

    # Test 5: Complete message generation
    print("TEST 5: Complete message generation")
    print("-" * 40)
    try:
        get_answer_prompt._system_prompt = None
        get_answer_prompt.update_chat_history(True)  # Use chat history mode
        get_answer_prompt.update_system_prompt(topic="DevOps")

        # Pre-filtered QA data (filtering happens outside module now)
        qa_data: list[get_answer_prompt.QAPair] = [
            {
                "question": "Docker benefits?",
                "answer": "Consistency, portability, isolation.",
                "similarity": 0.91,
            },
            {
                "question": "Service discovery?",
                "answer": "Consul, Kubernetes DNS.",
                "similarity": 0.88,
            },
        ]

        question = "How to scale containers?"

        # Generate complete message list
        messages = get_answer_prompt.get_messages(question, qa_data)
        save_messages(messages, "msgs_05_complete_generation", test_dir)

        # Validate structure
        assert len(messages) > 0, "No messages generated"
        assert messages[0]["role"] == "system", "First should be system"
        assert messages[-1]["role"] == "user", "Last should be user"

        # Count components
        system_count = sum(1 for m in messages if m["role"] == "system")
        user_count = sum(1 for m in messages if m["role"] == "user")
        assistant_count = sum(1 for m in messages if m["role"] == "assistant")

        # With chat history mode, we expect 2 QA pairs in history
        expected_history_pairs = 2
        expected_assistant = expected_history_pairs
        assert assistant_count == expected_assistant, (
            f"Expected {expected_assistant} assistant messages, got {assistant_count}"
        )

        # Verify all assistant messages have the new field
        for msg in messages:
            if msg["role"] == "assistant":
                assert '"sources_used_reasoning"' in msg["content"], (
                    "Assistant message should have 'sources_used_reasoning' field"
                )

        print(f"Generated {len(messages)} messages:")
        print(f"  - System: {system_count}")
        print(f"  - User: {user_count}")
        print(f"  - Assistant: {assistant_count}")

        tests_passed += 1

    except AssertionError as e:
        print(f"Failed: {e}")
        tests_failed += 1
        failed_tests.append("Test 5")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        tests_failed += 1
        failed_tests.append("Test 5")
    print()

    # Test 6: Error handling
    print("TEST 6: Error handling")
    print("-" * 40)
    test_6_passed = True

    # Sub-test 6a: Empty QA pairs
    try:
        print("  6a) Empty QA pairs")
        get_answer_prompt.update_chat_history(False)  # Use context mode
        messages = get_answer_prompt.get_messages("What is AI?", [])
        save_messages(messages, "msgs_06a_empty_qa", test_dir)

        assert len(messages) == 2, "Should have system + user"
        print(f"      Handled gracefully ({len(messages)} messages)")
    except Exception as e:
        print(f"      Failed: {e}")
        test_6_passed = False

    # Sub-test 6b: Optional similarity field
    try:
        print("  6b) Optional similarity field")
        qa_without_similarity: list[get_answer_prompt.QAPair] = [
            {"question": "Q1", "answer": "A1"},  # No similarity field
            {"question": "Q2", "answer": "A2", "similarity": 0.8},  # With similarity
        ]

        get_answer_prompt.update_chat_history(True)
        messages = get_answer_prompt.get_messages("Test", qa_without_similarity)

        # Both pairs should be in history (no filtering in module)
        expected = 1 + 4 + 1  # system + 2 pairs history + user
        assert len(messages) == expected, f"Expected {expected}, got {len(messages)}"

        save_messages(messages, "msgs_06b_optional_similarity", test_dir)
        print(f"      Optional similarity handled ({len(messages)} messages)")
    except Exception as e:
        print(f"      Failed: {e}")
        test_6_passed = False

    # Sub-test 6c: Missing required fields
    try:
        print("  6c) Missing required fields (via run())")
        # This should fail validation in run()
        bad_qa = [{"question": "Q1"}]  # Missing 'answer'

        result, messages, response = get_answer_prompt.run("Test?", bad_qa)
        result_data = json.loads(result)

        assert "error" in result_data, "Should return error for invalid QA structure"
        assert "answer" in result_data["error"], "Error should mention missing 'answer'"

        print("      Validation error caught correctly")
    except Exception as e:
        print(f"      Failed: {e}")
        test_6_passed = False

    if test_6_passed:
        print("Test 6 passed")
        tests_passed += 1
    else:
        print("Test 6 failed")
        tests_failed += 1
        failed_tests.append("Test 6")
    print()

    # Test 7: TypedDict compatibility
    print("TEST 7: TypedDict QAPair usage")
    print("-" * 40)
    try:
        # Create properly typed QA pairs
        typed_pairs: list[get_answer_prompt.QAPair] = [
            {"question": "Q1?", "answer": "A1", "similarity": 0.9},
            {"question": "Q2?", "answer": "A2"},  # Optional similarity
        ]

        get_answer_prompt.update_chat_history(False)  # Use context mode
        messages = get_answer_prompt.get_messages("New question?", typed_pairs)

        # Both pairs should be in context (no filtering)
        expected = 2  # system + user with context
        assert len(messages) == expected, f"Expected {expected}, got {len(messages)}"

        # Verify both pairs are in context
        user_content = messages[1]["content"]
        assert "Q1?" in user_content
        assert "Q2?" in user_content

        save_messages(messages, "msgs_07_typed_dict", test_dir)
        print("TypedDict handling correct")
        tests_passed += 1
    except Exception as e:
        print(f"Failed: {e}")
        tests_failed += 1
        failed_tests.append("Test 7")
    print()

    # Test 8: Context mode persistence
    print("TEST 8: Context mode persistence")
    print("-" * 40)
    try:
        # Set to history mode
        get_answer_prompt.update_chat_history(True)
        assert get_answer_prompt._use_chat_history == True, "History mode not set"

        # Generate messages
        qa: list[get_answer_prompt.QAPair] = [{"question": "Test Q", "answer": "Test A"}]
        msg1 = get_answer_prompt.get_messages("Q1?", qa)
        msg2 = get_answer_prompt.get_messages("Q2?", qa)

        # Both should use history mode
        assert len(msg1) == 4  # system + history pair + user
        assert len(msg2) == 4  # system + history pair + user

        # Switch to context mode
        get_answer_prompt.update_chat_history(False)
        assert get_answer_prompt._use_chat_history == False, "Context mode not set"

        msg3 = get_answer_prompt.get_messages("Q3?", qa)
        assert len(msg3) == 2  # system + user with context

        print("Mode persistence works correctly")
        tests_passed += 1
    except Exception as e:
        print(f"Failed: {e}")
        tests_failed += 1
        failed_tests.append("Test 8")
    print()

    # Test 9: Parsing JSON response with new field
    print("TEST 9: JSON response parsing with sources_used_reasoning")
    print("-" * 40)
    try:
        # Test valid JSON with new field
        valid_json = """
        {
            "answer": "Test answer content",
            "confidence": 0.85,
            "sources_used": ["context"],
            "sources_used_reasoning": "Using context from dialogue. Information comes directly from conversation history."
        }
        """

        result = get_answer_prompt._parse_json_response(valid_json)
        assert "sources_used_reasoning" in result, (
            "Missing 'sources_used_reasoning' in parsed result"
        )
        assert isinstance(result["sources_used_reasoning"], str), "Reasoning should be a string"
        assert len(result["sources_used_reasoning"]) > 0, "Reasoning should not be empty"

        print("      Valid JSON with reasoning parsed correctly")

        # Test fallback extraction for new field
        malformed = 'answer is "Test", confidence: 0.5, sources_used_reasoning: "Using general knowledge. No relevant context found."'
        fallback_result = get_answer_prompt._parse_json_response(malformed)
        assert "sources_used_reasoning" in fallback_result, "Fallback should handle reasoning field"

        print("      Fallback extraction handles reasoning field")

        tests_passed += 1
    except AssertionError as e:
        print(f"Failed: {e}")
        tests_failed += 1
        failed_tests.append("Test 9")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        tests_failed += 1
        failed_tests.append("Test 9")
    print()

    # Test 10: Full run() function with LLM call
    print("TEST 10: Full run() function with LLM call")
    print("-" * 40)

    test_10_passed = True

    # Sub-test 10a: With context QA pairs
    try:
        print("  10a) run() with context QA pairs")

        # Reset and configure
        get_answer_prompt._system_prompt = None
        get_answer_prompt.update_chat_history(True)  # Use chat history mode
        get_answer_prompt.update_system_prompt(topic="Python Programming")

        # Prepare QA pairs
        qa_pairs: list[get_answer_prompt.QAPair] = [
            {
                "question": "What are Python decorators?",
                "answer": "Functions that modify other functions.",
                "similarity": 0.9,
            },
            {
                "question": "How to handle errors in Python?",
                "answer": "Use try-except blocks with specific exception types.",
                "similarity": 0.85,
            },
        ]

        question = "What is a context manager in Python?"

        print(f"      Calling LLM with question: '{question}'")
        print(f"      Using {len(qa_pairs)} QA pairs as context")

        # Call run() with real LLM
        result_json, messages, raw_response = get_answer_prompt.run(
            user_question=question,
            qa_pairs=qa_pairs,
        )

        # Parse result
        result = json.loads(result_json)

        # Validate response structure
        assert "error" not in result, f"LLM call failed: {result.get('error')}"
        assert "answer" in result, "Missing 'answer' field in response"
        assert "confidence" in result, "Missing 'confidence' field in response"
        assert "sources_used" in result, "Missing 'sources_used' field in response"
        assert "sources_used_reasoning" in result, (
            "Missing 'sources_used_reasoning' field in response"
        )

        # Validate field types
        assert isinstance(result["answer"], str), "Answer should be string"
        assert isinstance(result["confidence"], (int, float)), "Confidence should be number"
        assert isinstance(result["sources_used"], list), "Sources should be list"
        assert isinstance(result["sources_used_reasoning"], str), "Reasoning should be string"

        # Validate confidence range
        assert 0 <= result["confidence"] <= 1, f"Confidence {result['confidence']} out of range"

        # Validate sources_used values
        valid_sources = {"context", "domain_knowledge"}
        for source in result["sources_used"]:
            assert source in valid_sources, f"Invalid source: {source}"

        # Check reasoning format (should have two sentences)
        reasoning = result["sources_used_reasoning"]
        # Count sentences by periods followed by space or end
        sentence_count = len([s for s in reasoning.split(". ") if s])

        # Save result
        result_file = test_dir / "result_10a_run_with_context.json"
        result_file.write_text(result_json, encoding="utf-8")

        print("      LLM response received successfully")
        print(f"      Answer length: {len(result['answer'])} chars")
        print(f"      Confidence: {result['confidence']}")
        print(f"      Sources: {result['sources_used']}")
        print(f"      Reasoning sentences: ~{sentence_count}")
        print(f"      Result saved to: {result_file.name}")

        # Save messages sent to LLM
        save_messages(messages, "msgs_10a_run_with_context", test_dir)

    except Exception as e:
        print(f"      Failed: {e}")
        test_10_passed = False

    # Sub-test 10b: Without context (domain knowledge only)
    try:
        print("  10b) run() without context (domain knowledge only)")

        # Reset for domain-only test
        get_answer_prompt._system_prompt = None
        get_answer_prompt.update_chat_history(False)
        get_answer_prompt.update_system_prompt(topic="Mathematics")

        question = "What is the Pythagorean theorem?"

        print(f"      Calling LLM with question: '{question}'")
        print("      Using no QA pairs (domain knowledge only)")

        # Call run() with empty QA pairs
        result_json, messages, raw_response = get_answer_prompt.run(
            user_question=question,
            qa_pairs=[],  # Empty context
        )

        # Parse result
        result = json.loads(result_json)

        # Validate response
        assert "error" not in result, f"LLM call failed: {result.get('error')}"
        assert "sources_used_reasoning" in result, "Missing reasoning field"

        # For domain-only questions, should use domain_knowledge
        if "domain_knowledge" in result["sources_used"]:
            print("      Correctly identified as domain knowledge")

        # Save result
        result_file = test_dir / "result_10b_run_domain_only.json"
        result_file.write_text(result_json, encoding="utf-8")

        print("      LLM response received successfully")
        print(f"      Confidence: {result['confidence']}")
        print(f"      Sources: {result['sources_used']}")
        print(f"      Result saved to: {result_file.name}")

        # Save messages sent to LLM
        save_messages(messages, "msgs_10b_run_domain_only", test_dir)

    except Exception as e:
        print(f"      Failed: {e}")
        test_10_passed = False

    # Sub-test 10c: Context mode switching with run()
    try:
        print("  10c) run() with context mode (not history)")

        # Switch to context mode
        get_answer_prompt.update_chat_history(False)  # Use context mode, not history

        qa_pairs: list[get_answer_prompt.QAPair] = [
            {
                "question": "What is REST API?",
                "answer": "Architectural style for web services using HTTP.",
                "similarity": 0.88,
            }
        ]

        question = "How to design APIs?"

        print("      Calling LLM in context mode")

        # Call run() in context mode
        result_json, messages, raw_response = get_answer_prompt.run(
            user_question=question,
            qa_pairs=qa_pairs,
        )

        result = json.loads(result_json)

        # Validate
        assert "error" not in result, f"LLM call failed: {result.get('error')}"
        assert "sources_used_reasoning" in result, "Missing reasoning field"

        # Save result
        result_file = test_dir / "result_10c_run_context_mode.json"
        result_file.write_text(result_json, encoding="utf-8")

        print("      Context mode run successful")
        print(f"      Result saved to: {result_file.name}")

        # Save messages sent to LLM
        save_messages(messages, "msgs_10c_run_context_mode", test_dir)

    except Exception as e:
        print(f"      Failed: {e}")
        test_10_passed = False

    if test_10_passed:
        print("Test 10 passed")
        tests_passed += 1
    else:
        print("Test 10 failed")
        tests_failed += 1
        failed_tests.append("Test 10")
    print()

    # Final summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = tests_passed + tests_failed
    print(f"Tests passed: {tests_passed}/{total}")

    # List all saved files
    saved_files = sorted(test_dir.glob("msgs_*.txt"))
    result_files = sorted(test_dir.glob("result_*.json"))

    if saved_files:
        print(f"\nGenerated {len(saved_files)} message files:")
        for f in saved_files:
            size = f.stat().st_size
            print(f"   {f.name:<45} {size:>7,} bytes")

    if result_files:
        print(f"\nGenerated {len(result_files)} result files:")
        for f in result_files:
            size = f.stat().st_size
            print(f"   {f.name:<45} {size:>7,} bytes")

    print("\n" + "=" * 60)

    if tests_failed > 0:
        print("TESTS FAILED")
        print(f"  Failed: {', '.join(failed_tests)}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        print(f"  Check {test_dir}/ for message analysis and LLM results")
        sys.exit(0)


if __name__ == "__main__":
    run_tests()
