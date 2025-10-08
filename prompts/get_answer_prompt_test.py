"""Test module for get_answer_prompt with proper imports.

This test module validates the answer generation prompt functionality,
including similarity-based filtering, message generation, and error handling.
Saves all generated messages to JSON files for analysis.
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

    # Create test output directory next to the test script
    test_script_dir = Path(__file__).parent
    test_dir = test_script_dir / "test_messages"
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

    # Test 4: Similarity threshold filtering
    print("TEST 4: Similarity threshold filtering")
    print("-" * 40)

    # Reset for clean state
    get_answer_prompt._system_prompt = None
    get_answer_prompt._similarity_threshold = 2.0
    get_answer_prompt.update_system_prompt(topic="Machine Learning")

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
        {
            "question": "Database choices?",
            "answer": "PostgreSQL, MongoDB, Redis.",
            "similarity": 0.45,
        },
    ]

    test_4_passed = True

    # Sub-test 4a: Threshold 0.8 (filters to history)
    try:
        print("  4a) Threshold=0.8 (includes 2 pairs in history)")
        get_answer_prompt.update_similarity_threshold(0.8)

        messages = get_answer_prompt.get_messages("How to improve models?", qa_pairs)

        # Should have: 1 system + 4 history (2 pairs * 2 messages) + 1 user
        expected = 1 + 4 + 1
        assert len(messages) == expected, f"Expected {expected}, got {len(messages)}"

        # Verify history messages contain JSON responses
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

        save_messages(messages, "msgs_04a_threshold_0.8", test_dir)
        print(f"      {len(messages)} messages generated (with history)")
    except AssertionError as e:
        print(f"      Failed: {e}")
        test_4_passed = False

    # Sub-test 4b: Threshold 0.95 (excludes all)
    try:
        print("  4b) Threshold=0.95 (no pairs qualify)")
        get_answer_prompt.update_similarity_threshold(0.95)

        messages = get_answer_prompt.get_messages("Test question", qa_pairs)

        # Should have only system + user (no history)
        assert len(messages) == 2, f"Expected 2, got {len(messages)}"
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        save_messages(messages, "msgs_04b_threshold_0.95", test_dir)
        print("      No history messages (as expected)")
    except AssertionError as e:
        print(f"      Failed: {e}")
        test_4_passed = False

    # Sub-test 4c: Threshold > 1.0 (all to context)
    try:
        print("  4c) Threshold=1.5 (all pairs to context)")
        get_answer_prompt.update_similarity_threshold(1.5)

        messages = get_answer_prompt.get_messages("How to improve models?", qa_pairs)

        # Should have only system + user
        assert len(messages) == 2, f"Expected 2 messages (system+user), got {len(messages)}"
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        # Verify context is in user message
        user_content = messages[1]["content"]
        assert "CONTEXT Q&A PAIRS" in user_content, "Context should be in user message"
        # All 3 pairs should be present
        assert "transfer learning" in user_content
        assert "overfitting" in user_content
        assert "Database choices" in user_content

        save_messages(messages, "msgs_04c_threshold_above_1", test_dir)
        print("      All pairs in context (no history)")
    except AssertionError as e:
        print(f"      Failed: {e}")
        test_4_passed = False

    # Sub-test 4d: Threshold exactly 1.0
    try:
        print("  4d) Threshold=1.0 (boundary case)")
        get_answer_prompt.update_similarity_threshold(1.0)

        messages = get_answer_prompt.get_messages("Test question", qa_pairs)

        # No pairs have similarity >= 1.0, so no history
        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"

        save_messages(messages, "msgs_04d_threshold_1.0", test_dir)
        print("      No history (boundary excluded)")
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
        get_answer_prompt.update_similarity_threshold(0.8)
        get_answer_prompt.update_system_prompt(topic="DevOps")

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
            {
                "question": "Database migrations?",
                "answer": "Flyway, Liquibase.",
                "similarity": 0.65,
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

        # With threshold 0.8, we expect 2 QA pairs in history
        expected_history_pairs = 2
        expected_assistant = expected_history_pairs
        assert assistant_count == expected_assistant, (
            f"Expected {expected_assistant} assistant messages, got {assistant_count}"
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
        messages = get_answer_prompt.get_messages("What is AI?", [])
        save_messages(messages, "msgs_06a_empty_qa", test_dir)

        assert len(messages) == 2, "Should have system + user"
        print(f"      Handled gracefully ({len(messages)} messages)")
    except Exception as e:
        print(f"      Failed: {e}")
        test_6_passed = False

    # Sub-test 6b: Invalid/missing similarity
    try:
        print("  6b) Invalid/missing similarity values")
        invalid_qa: list[Any] = [
            {"question": "Q1", "answer": "A1", "similarity": "invalid"},  # Invalid type
            {"question": "Q2", "answer": "A2", "similarity": 0.8},  # Valid
            {"question": "Q3", "answer": "A3"},  # Missing similarity
        ]

        get_answer_prompt.update_similarity_threshold(0.7)
        messages = get_answer_prompt.get_messages("Test", invalid_qa)

        # Only Q2 should be in history (similarity 0.8 >= threshold 0.7)
        # Invalid and missing similarities treated as 0.0
        expected = 1 + 2 + 1  # system + 1 pair history + user
        assert len(messages) == expected, f"Expected {expected}, got {len(messages)}"

        save_messages(messages, "msgs_06b_invalid_similarity", test_dir)
        print(f"      Handled invalid data ({len(messages)} messages)")
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

        get_answer_prompt.update_similarity_threshold(0.85)
        messages = get_answer_prompt.get_messages("New question?", typed_pairs)

        # Only first pair should be in history
        expected = 1 + 2 + 1  # system + 1 pair + user
        assert len(messages) == expected, f"Expected {expected}, got {len(messages)}"

        save_messages(messages, "msgs_07_typed_dict", test_dir)
        print("TypedDict handling correct")
        tests_passed += 1
    except Exception as e:
        print(f"Failed: {e}")
        tests_failed += 1
        failed_tests.append("Test 7")
    print()

    # Final summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = tests_passed + tests_failed
    print(f"Tests passed: {tests_passed}/{total}")

    # List all saved files
    saved_files = sorted(test_dir.glob("msgs_*.txt"))
    if saved_files:
        print(f"\nGenerated {len(saved_files)} message files:")
        for f in saved_files:
            size = f.stat().st_size
            print(f"   {f.name:<45} {size:>7,} bytes")

    print("\n" + "=" * 60)

    if tests_failed > 0:
        print("TESTS FAILED")
        print(f"  Failed: {', '.join(failed_tests)}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        print(f"  Check {test_dir}/ for message analysis")
        sys.exit(0)


if __name__ == "__main__":
    run_tests()
