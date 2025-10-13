"""Test module for get_category_prompt with message saving.

This test module validates the question categorization prompt functionality,
including category initialization, question classification with/without answers,
confidence scoring, and error handling. Saves all generated messages to files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any


# Add parent directory to sys.path if needed
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import the module to test
import get_category_prompt


def save_messages(messages: list[dict[str, str]], filename: str, test_dir: Path) -> None:
    """Save messages to formatted text file with real newlines.

    Args:
        messages: List of message dictionaries to save.
        filename: Name for the output file (without extension).
        test_dir: Directory to save files in.
    """
    txt_path = test_dir / f"{filename}.txt"

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
            escaped = obj.replace("\\", "\\\\").replace('"', '\\"')
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
    print("TEST MODULE FOR get_category_prompt.py")
    print("=" * 60)
    print(f"Working directory: {Path.cwd()}")
    print(f"Test module path: {Path(__file__)}")
    print("=" * 60 + "\n")

    # Create test output directory
    test_script_dir = Path(__file__).parent
    test_module_name = Path(__file__).stem
    test_dir = test_script_dir / test_module_name
    test_dir.mkdir(exist_ok=True)
    print(f"Output directory: {test_dir.absolute()}\n")

    # Track results
    tests_passed = 0
    tests_failed = 0
    failed_tests = []

    # Test 1: Initialize with categories
    print("TEST 1: Initialize categorization system")
    print("-" * 40)
    try:
        # Clear any previous state
        get_category_prompt._system_prompt = None
        get_category_prompt._chat_history = []
        get_category_prompt._current_categories = None

        test_categories = {
            "Technical": "Questions about implementation, code, and architecture",
            "Business": "Questions about ROI, costs, and business value",
            "Process": "Questions about workflows and methodologies",
            "Security": "Questions about security, authentication, and compliance",
        }

        prompt = get_category_prompt.update_system_prompt(categories=test_categories)
        assert prompt is not None, "Prompt is None"
        assert len(prompt) > 0, "Prompt is empty"
        assert isinstance(prompt, str), f"Prompt is {type(prompt)}, not str"

        # Verify categories are in prompt
        for category in test_categories:
            assert category in prompt, f"Category '{category}' not in prompt"

        # Save the system prompt
        init_msg = [{"role": "system", "content": prompt}]
        save_messages(init_msg, "msgs_01_init_categories", test_dir)

        print(f"Initialized: {len(prompt):,} characters")
        print(f"Categories: {', '.join(test_categories.keys())}")
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

    # Test 2: Categorization without answer context
    print("TEST 2: Categorize questions WITHOUT answer context")
    print("-" * 40)
    try:
        # Ensure categories are initialized and history is clean
        get_category_prompt.update_system_prompt(categories=test_categories)
        get_category_prompt.update_chat_history(clear=True)

        test_questions = [
            "How do we implement OAuth2 authentication?",
            "What's the ROI of cloud migration?",
            "How to set up CI/CD pipeline?",
        ]

        for i, question in enumerate(test_questions, 1):
            print(f"  Question {i}: {question[:50]}...")

            # Get messages that would be sent
            messages = get_category_prompt.get_messages(question)
            save_messages(messages, f"msgs_02_no_answer_{i}", test_dir)

            assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert question in messages[1]["content"]

            print(f"    Generated {len(messages)} messages")

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

    # Test 3: Categorization with answer context
    print("TEST 3: Categorize questions WITH answer context")
    print("-" * 40)
    try:
        # Ensure clean state
        get_category_prompt.update_chat_history(clear=True)

        test_qa_pairs = [
            {
                "question": "How to handle database migrations?",
                "answer": "We use Flyway for version control and automated migrations.",
            },
            {
                "question": "What's our budget for the project?",
                "answer": "The allocated budget is $500K for the first phase.",
            },
        ]

        for i, qa in enumerate(test_qa_pairs, 1):
            print(f"  Q{i}: {qa['question'][:40]}...")
            print(f"  A{i}: {qa['answer'][:40]}...")

            # Get messages with answer context
            messages = get_category_prompt.get_messages(qa["question"], qa["answer"])
            save_messages(messages, f"msgs_03_with_answer_{i}", test_dir)

            assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"

            # Verify answer is included in user message
            user_content = messages[1]["content"]
            assert "ANSWER:" in user_content, "Answer marker should be present"
            assert qa["answer"] in user_content, "Answer content should be included"

            print("    Answer context included")

        tests_passed += 1
    except AssertionError as e:
        print(f"Failed: {e}")
        tests_failed += 1
        failed_tests.append("Test 3")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        tests_failed += 1
        failed_tests.append("Test 3")
    print()

    # Test 4: Chat history functionality
    print("TEST 4: Chat history management")
    print("-" * 40)
    test_4_passed = True

    # Sub-test 4a: Clear history
    try:
        print("  4a) Clear chat history")
        get_category_prompt.update_chat_history(clear=True)
        history = get_category_prompt.update_chat_history()
        assert len(history) == 0, f"Expected empty history, got {len(history)} items"
        print("      History cleared successfully")
    except AssertionError as e:
        print(f"      Failed: {e}")
        test_4_passed = False

    # Sub-test 4b: Add to history
    try:
        print("  4b) Add messages to history")
        get_category_prompt.add_to_chat_history({"role": "user", "content": "Test question"})
        get_category_prompt.add_to_chat_history(
            {"role": "assistant", "content": '{"category": "Technical"}'}
        )

        history = get_category_prompt.update_chat_history()
        assert len(history) == 2, f"Expected 2 messages, got {len(history)}"

        # Generate messages with history
        messages = get_category_prompt.get_messages("New question")
        save_messages(messages, "msgs_04_with_history", test_dir)

        # Should have system + history + user
        assert len(messages) == 4, f"Expected 4 messages, got {len(messages)}"
        print("      Added 2 messages to history")
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

    # Test 5: Category validation and caching
    print("TEST 5: Category validation and caching")
    print("-" * 40)
    try:
        # Clear history before changing categories
        get_category_prompt.update_chat_history(clear=True)

        # Set specific categories
        validation_categories = {
            "Frontend": "UI/UX related questions",
            "Backend": "Server-side questions",
            "DevOps": "Deployment and operations",
        }

        get_category_prompt.update_system_prompt(categories=validation_categories)

        # Verify categories are cached
        assert get_category_prompt._current_categories is not None
        assert set(get_category_prompt._current_categories.keys()) == set(
            validation_categories.keys()
        )

        # Generate messages for validation test
        messages = get_category_prompt.get_messages("How to deploy to AWS?")
        save_messages(messages, "msgs_05_validation", test_dir)

        print(f"Categories cached: {', '.join(validation_categories.keys())}")
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

    # Sub-test 6a: Uninitialized system prompt
    try:
        print("  6a) Uninitialized system prompt")
        # Reset system prompt
        get_category_prompt._system_prompt = None

        try:
            messages = get_category_prompt.get_messages("Test")
            print("      Should have raised ValueError")
            test_6_passed = False
        except ValueError as e:
            assert "not initialized" in str(e)
            print("      Correctly raised ValueError")
    except Exception as e:
        print(f"      Failed unexpectedly: {e}")
        test_6_passed = False

    # Sub-test 6b: Invalid categories
    try:
        print("  6b) Invalid categories")
        try:
            get_category_prompt.update_system_prompt(categories={})
            print("      Should have raised ValueError for empty categories")
            test_6_passed = False
        except ValueError:
            # Just check that ValueError was raised, don't check specific text
            print("      Correctly rejected empty categories")
    except Exception as e:
        print(f"      Failed unexpectedly: {e}")
        test_6_passed = False

    # Sub-test 6c: Invalid message for history
    try:
        print("  6c) Invalid message for history")
        try:
            get_category_prompt.add_to_chat_history("not a dict")
            print("      Should have raised ValueError")
            test_6_passed = False
        except ValueError as e:
            assert "dictionary" in str(e)
            print("      Correctly rejected invalid message")
    except Exception as e:
        print(f"      Failed unexpectedly: {e}")
        test_6_passed = False

    if test_6_passed:
        print("Test 6 passed")
        tests_passed += 1
    else:
        print("Test 6 failed")
        tests_failed += 1
        failed_tests.append("Test 6")
    print()

    # Test 7: Complete workflow simulation
    print("TEST 7: Complete categorization workflow")
    print("-" * 40)
    try:
        # Reset and initialize fresh
        get_category_prompt._system_prompt = None
        get_category_prompt._chat_history = []

        workflow_categories = {
            "Infrastructure": "Cloud, servers, and deployment",
            "Development": "Coding, testing, and debugging",
            "Management": "Project planning and team coordination",
        }

        get_category_prompt.update_system_prompt(categories=workflow_categories)

        # Simulate a sequence of categorizations
        questions = [
            ("How to set up Kubernetes?", None),
            ("What's the sprint velocity?", None),
            ("Debug memory leak?", "Check heap dumps and profiling tools"),
        ]

        for i, (q, a) in enumerate(questions, 1):
            messages = get_category_prompt.get_messages(q, a)
            save_messages(messages, f"msgs_07_workflow_{i}", test_dir)

            print(f"  Step {i}: {len(messages)} messages")
            if a:
                print("    (with answer context)")

        print("Workflow completed successfully")
        tests_passed += 1
    except Exception as e:
        print(f"Failed: {e}")
        tests_failed += 1
        failed_tests.append("Test 7")
    print()

    # Test 8: run() function and JSON parsing
    print("TEST 8: run() function and JSON parsing")
    print("-" * 40)
    test_8_passed = True

    # Sub-test 8a: Test JSON parsing helper functions
    try:
        print("  8a) Test JSON parsing functions")

        # Test valid JSON extraction
        test_responses = [
            (
                '{"category": "Technical", "confidence": 0.95, "reasoning": "Test reason"}',
                {"category": "Technical", "confidence": 0.95, "reasoning": "Test reason"},
            ),
            (
                'Some text before {"category": "Business", "confidence": 0.7, "reasoning": "Business logic"}',
                {"category": "Business", "confidence": 0.7, "reasoning": "Business logic"},
            ),
            (
                'category: "Process", confidence: 0.5, reasoning: "Process flow"',
                {"category": "Process", "confidence": 0.5, "reasoning": "Process flow"},
            ),
        ]

        # Set categories for validation
        get_category_prompt._current_categories = {
            "Technical": "Tech questions",
            "Business": "Business questions",
            "Process": "Process questions",
        }

        for response_text, expected_keys in test_responses:
            result = get_category_prompt._parse_json_response(response_text)
            assert "category" in result, "Missing 'category' in parsed result"
            assert "confidence" in result, "Missing 'confidence' in parsed result"
            assert "reasoning" in result, "Missing 'reasoning' in parsed result"
            print(f"      Parsed: {result['category']} ({result['confidence']})")

        print("      JSON parsing functions work correctly")
    except Exception as e:
        print(f"      Failed: {e}")
        test_8_passed = False

    # Sub-test 8b: Test run() with uninitialized system
    try:
        print("  8b) Test run() with uninitialized system")
        get_category_prompt._system_prompt = None

        result_json, messages, response = get_category_prompt.run("Test question")
        result = json.loads(result_json)

        assert "error" in result, "Should return error for uninitialized system"
        assert "not initialized" in result["error"].lower()
        print("      Correctly handled uninitialized system")
    except Exception as e:
        print(f"      Failed: {e}")
        test_8_passed = False

    # Sub-test 8c: Test run() with real API (will fail gracefully if no API)
    try:
        print("  8c) Test run() with real API call")

        # Initialize system
        categories = {
            "Technical": "Implementation and coding questions",
            "Business": "ROI and business value questions",
        }
        get_category_prompt.update_system_prompt(categories=categories)
        get_category_prompt.update_chat_history(clear=True)

        # Try to run with real API
        question = "How to implement OAuth2?"
        answer = "Use authorization code flow with PKCE"

        result_json, messages, response = get_category_prompt.run(question, answer=answer)
        result = json.loads(result_json)

        # Save the messages and result
        save_messages(messages, "msgs_08_run_test", test_dir)

        # Check result structure
        if "error" in result:
            # API call failed (expected if no API key)
            print(f"      API call failed (expected): {result['error'][:50]}...")
            assert len(messages) > 0, "Messages should be generated even on API error"
        else:
            # API call succeeded
            assert "category" in result, "Missing 'category' in result"
            assert "confidence" in result, "Missing 'confidence' in result"
            assert "reasoning" in result, "Missing 'reasoning' in result"

            # Validate category is one of the defined ones
            assert result["category"] in categories, f"Invalid category: {result['category']}"

            # Validate confidence range
            assert 0 <= result["confidence"] <= 1, f"Invalid confidence: {result['confidence']}"

            print(f"      API call succeeded: {result['category']} ({result['confidence']})")

        # Verify messages structure
        assert len(messages) >= 2, f"Expected at least 2 messages, got {len(messages)}"
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert question in messages[-1]["content"]
        if answer:
            assert answer in messages[-1]["content"]

        print("      run() function tested successfully")
    except Exception as e:
        print(f"      Failed: {e}")
        test_8_passed = False

    # Sub-test 8d: Test fallback extraction functions
    try:
        print("  8d) Test fallback extraction functions")

        # Test category extraction
        test_texts = [
            ('"category": "Technical"', "Technical"),
            ("category: 'Business'", "Business"),
            ("category is Process", "Process"),
        ]

        for text, expected in test_texts:
            result = get_category_prompt._extract_category(text)
            assert result == expected, f"Expected {expected}, got {result}"

        # Test confidence extraction
        conf_result = get_category_prompt._extract_confidence('"confidence": 0.85')
        assert conf_result == 0.85, f"Expected 0.85, got {conf_result}"

        # Test reasoning extraction
        reason_result = get_category_prompt._extract_reasoning('"reasoning": "Test reason"')
        assert reason_result == "Test reason", f"Expected 'Test reason', got {reason_result}"

        print("      Fallback extraction functions work correctly")
    except Exception as e:
        print(f"      Failed: {e}")
        test_8_passed = False

    if test_8_passed:
        print("Test 8 passed")
        tests_passed += 1
    else:
        print("Test 8 failed")
        tests_failed += 1
        failed_tests.append("Test 8")
    print()

    # Final summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = tests_passed + tests_failed
    print(f"Tests passed: {tests_passed}/{total}")

    if tests_failed > 0:
        print(f"Tests failed: {', '.join(failed_tests)}")

    # List saved files
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
