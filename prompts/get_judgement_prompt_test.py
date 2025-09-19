"""Test module for symmetric entailment judgement functionality.

This module provides comprehensive testing for the get_judgement_prompt module,
verifying correct implementation of bidirectional semantic entailment evaluation.

Tests cover precision/recall scoring, contradiction/hallucination detection,
numerical tolerance, and edge cases as specified in the requirements.

Example:
    Run all tests:

    ```bash
    python -m prompts.get_judgement_prompt_test
    ```

    Or use pytest for more detailed output:

    ```bash
    pytest prompts/get_judgement_prompt_test.py -v
    ```
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock, Mock, patch

from utils.logger import get_logger

from prompts import get_judgement_prompt


logger = get_logger(__name__)


class TestResult:
    """Container for test result tracking.

    Attributes:
        name: Test name/description.
        passed: Whether test passed all validations.
        errors: List of validation error messages.
    """

    def __init__(self, name: str) -> None:
        """Initialize test result.

        Args:
            name: Test name/description.
        """
        self.name = name
        self.passed = True
        self.errors: list[str] = []

    def fail(self, message: str) -> None:
        """Mark test as failed with error message.

        Args:
            message: Error description.
        """
        self.passed = False
        self.errors.append(message)


def test_initialization() -> TestResult:
    """Test system initialization.

    Returns:
        TestResult with validation status.
    """
    result = TestResult("System Initialization")

    try:
        # Initialize without parameters
        prompt = get_judgement_prompt.update_system_prompt()

        if not prompt:
            result.fail("System prompt is empty")

        if len(prompt) < 1000:
            result.fail(f"System prompt too short: {len(prompt)} chars")

        logger.info(f"System prompt initialized: {len(prompt)} characters")

    except Exception as e:
        result.fail(f"Initialization failed: {e}")
        logger.error(f"Initialization error: {e}")

    return result


def test_perfect_match() -> TestResult:
    """Test evaluation of semantically identical answers.

    Returns:
        TestResult with validation status.
    """
    result = TestResult("Perfect Match Evaluation")

    question = "What is the capital of France?"
    reference = "The capital of France is Paris."
    candidate = "Paris is the capital of France."

    try:
        result_json, messages, response = get_judgement_prompt.run(question, reference, candidate)
        output = json.loads(result_json)

        # Validate high scores for perfect match
        if output["score"] < get_judgement_prompt.THRESHOLD_GOOD:
            result.fail(
                f"Score {output['score']} should be >= {get_judgement_prompt.THRESHOLD_GOOD}"
            )

        if output["class"] != "good":
            result.fail(f"Class should be 'good', got '{output['class']}'")

        if output["contradiction"]:
            result.fail("Should not have contradiction for perfect match")

        if output["hallucination"]:
            result.fail("Should not have hallucination for perfect match")

        logger.info(f"Perfect match score: {output['score']}")

    except Exception as e:
        result.fail(f"Exception: {e}")
        logger.error(f"Perfect match test error: {e}")

    return result


def test_partial_match() -> TestResult:
    """Test evaluation with missing information.

    Returns:
        TestResult with validation status.
    """
    result = TestResult("Partial Match with Missing Information")

    question = "What are the key features of Docker?"
    reference = "Docker provides containerization, portability, isolation, and resource efficiency."
    candidate = "Docker provides containerization and portability."

    try:
        result_json, messages, response = get_judgement_prompt.run(question, reference, candidate)
        output = json.loads(result_json)

        # Precision should be high (all candidate info is correct)
        if output["precision_c_to_r"] < 0.9:
            result.fail(
                f"Precision {output['precision_c_to_r']:.2f} too low for correct partial info"
            )

        # Recall should be moderate (covers only 2 of 4 features)
        if output["recall_r_to_c"] > 0.7:
            result.fail(f"Recall {output['recall_r_to_c']:.2f} too high for 50% coverage")

        if output["contradiction"]:
            result.fail("Should not have contradiction for partial but correct match")

        logger.info(
            f"Partial match P={output['precision_c_to_r']:.2f}, R={output['recall_r_to_c']:.2f}"
        )

    except Exception as e:
        result.fail(f"Exception: {e}")
        logger.error(f"Partial match test error: {e}")

    return result


def test_contradiction_detection() -> TestResult:
    """Test detection of contradictory information.

    Returns:
        TestResult with validation status.
    """
    result = TestResult("Contradiction Detection")

    question = "What is the speed of light?"
    reference = "The speed of light is approximately 300,000 km/s."
    candidate = "The speed of light is approximately 150,000 km/s."

    try:
        result_json, messages, response = get_judgement_prompt.run(question, reference, candidate)
        output = json.loads(result_json)

        if not output["contradiction"]:
            result.fail("Should detect contradiction (50% speed difference)")

        if output["penalties"] < get_judgement_prompt.CONTRADICTION_PENALTY:
            result.fail(
                f"Penalties {output['penalties']} should include "
                f"contradiction penalty {get_judgement_prompt.CONTRADICTION_PENALTY}"
            )

        if output["class"] == "good":
            result.fail("Class should not be 'good' with major contradiction")

        logger.info(
            f"Contradiction detected: {output['contradiction']}, penalties: {output['penalties']}"
        )

    except Exception as e:
        result.fail(f"Exception: {e}")
        logger.error(f"Contradiction test error: {e}")

    return result


def test_hallucination_detection() -> TestResult:
    """Test detection of hallucinated information.

    Returns:
        TestResult with validation status.
    """
    result = TestResult("Hallucination Detection")

    question = "What is Python?"
    reference = "Python is a high-level programming language."
    candidate = (
        "Python is a high-level programming language created by "
        "Guido van Rossum in 1991 with version 3.12 being the latest."
    )

    try:
        result_json, messages, response = get_judgement_prompt.run(question, reference, candidate)
        output = json.loads(result_json)

        if not output["hallucination"]:
            result.fail("Should detect hallucination (new facts: creator, year, version)")

        if (
            output["hallucination"]
            and output["penalties"] < get_judgement_prompt.HALLUCINATION_PENALTY
        ):
            result.fail(
                f"Penalties {output['penalties']} should include "
                f"hallucination penalty {get_judgement_prompt.HALLUCINATION_PENALTY}"
            )

        logger.info(f"Hallucination detected: {output['hallucination']}")

    except Exception as e:
        result.fail(f"Exception: {e}")
        logger.error(f"Hallucination test error: {e}")

    return result


def test_empty_candidate() -> TestResult:
    """Test edge case with empty candidate answer.

    Returns:
        TestResult with validation status per §8 specification.
    """
    result = TestResult("Empty Candidate Edge Case")

    question = "What is AI?"
    reference = "AI is artificial intelligence."
    candidate = ""

    try:
        result_json, messages, response = get_judgement_prompt.run(question, reference, candidate)
        output = json.loads(result_json)

        # Validate per §8 specification
        if output["score"] != 0:
            result.fail(f"Score should be 0 for empty candidate, got {output['score']}")

        if output["class"] != "bad":
            result.fail(f"Class should be 'bad' for empty candidate, got '{output['class']}'")

        if output["precision_c_to_r"] != 1.0:
            result.fail(
                f"Precision should be 1.0 (empty has no incorrect content), "
                f"got {output['precision_c_to_r']}"
            )

        if output["recall_r_to_c"] != 0.0:
            result.fail(
                f"Recall should be 0.0 (empty covers nothing), got {output['recall_r_to_c']}"
            )

        logger.info("Empty candidate handled correctly")

    except Exception as e:
        result.fail(f"Exception: {e}")
        logger.error(f"Empty candidate test error: {e}")

    return result


def test_numerical_tolerance() -> TestResult:
    """Test numerical tolerance within acceptable range.

    Returns:
        TestResult with validation status.
    """
    result = TestResult("Numerical Tolerance Check")

    question = "What is the measurement?"
    reference = "The measurement is 100.0 meters."
    candidate = "The measurement is 101.5 meters."  # 1.5% difference

    try:
        result_json, messages, response = get_judgement_prompt.run(question, reference, candidate)
        output = json.loads(result_json)

        tolerance_pct = get_judgement_prompt.NUMERICAL_TOLERANCE_RELATIVE * 100

        if output["contradiction"]:
            result.fail(f"Should NOT have contradiction (1.5% < {tolerance_pct}% tolerance)")

        if output["penalties"] >= get_judgement_prompt.CONTRADICTION_PENALTY:
            result.fail("Should not have contradiction penalty for value within tolerance")

        logger.info(
            f"Tolerance check: contradiction={output['contradiction']}, "
            f"diff=1.5%, tolerance={tolerance_pct}%"
        )

    except Exception as e:
        result.fail(f"Exception: {e}")
        logger.error(f"Numerical tolerance test error: {e}")

    return result


# NEW TESTS START HERE


def test_invalid_json_response() -> TestResult:
    """Test handling of invalid JSON responses from LLM.

    Returns:
        TestResult with validation status.
    """
    result = TestResult("Invalid JSON Response Handling")

    try:
        # Test completely invalid JSON
        invalid_response = "This is not JSON at all!"
        parsed = get_judgement_prompt._parse_json_response(invalid_response)

        # Should return valid dict with defaults from fallback
        required_keys = {
            "precision_c_to_r",
            "recall_r_to_c",
            "contradiction",
            "hallucination",
            "justification",
            "evidence",
        }
        if not all(key in parsed for key in required_keys):
            result.fail("Fallback should provide all required fields")

        # Test broken JSON with partial fields
        broken_json = '{"precision_c_to_r": 0.8, "recall_r_to_c": broken}'
        parsed = get_judgement_prompt._parse_json_response(broken_json)
        if not isinstance(parsed, dict):
            result.fail("Should return dict even with broken JSON")

        logger.info("Invalid JSON handled correctly via fallback")

    except Exception as e:
        result.fail(f"Exception: {e}")

    return result


def test_incomplete_llm_response() -> TestResult:
    """Test handling of incomplete responses with missing fields.

    Returns:
        TestResult with validation status.
    """
    result = TestResult("Incomplete LLM Response")

    try:
        # Missing required fields
        incomplete = '{"precision_c_to_r": 0.5}'
        parsed = get_judgement_prompt._parse_json_response(incomplete)

        if "recall_r_to_c" not in parsed:
            result.fail("Should provide recall_r_to_c via fallback")

        # Wrong data types
        wrong_types = '{"precision_c_to_r": "not_a_number", "recall_r_to_c": 0.5, "contradiction": "yes", "hallucination": false, "justification": 123, "evidence": "not_array"}'
        parsed = get_judgement_prompt._parse_json_response(wrong_types)

        if not isinstance(parsed["evidence"], list):
            result.fail("Should convert evidence to list")

        # Values out of bounds
        out_of_bounds = '{"precision_c_to_r": 1.5, "recall_r_to_c": -0.2, "contradiction": false, "hallucination": false, "justification": "test", "evidence": []}'
        parsed = get_judgement_prompt._parse_json_response(out_of_bounds)

        if not (0 <= parsed["precision_c_to_r"] <= 1):
            result.fail("Should clamp precision to [0,1]")
        if not (0 <= parsed["recall_r_to_c"] <= 1):
            result.fail("Should clamp recall to [0,1]")

        logger.info("Incomplete responses handled correctly")

    except Exception as e:
        result.fail(f"Exception: {e}")

    return result


def test_empty_reference() -> TestResult:
    """Test empty reference answer per §8 specification.

    Returns:
        TestResult with validation status.
    """
    result = TestResult("Empty Reference Answer")

    question = "What is AI?"
    reference = ""
    candidate = "AI is artificial intelligence."

    try:
        with patch("prompts.get_judgement_prompt.logger") as mock_logger:
            try:
                get_judgement_prompt.run(question, reference, candidate)
                result.fail("Should raise ValueError for empty reference")
            except ValueError as e:
                if "reference_answer must be non-empty" not in str(e):
                    result.fail(f"Wrong error message: {e}")
                # Check warning was logged
                mock_logger.warning.assert_called_with(
                    "Empty reference_answer - exclude from aggregates"
                )

        logger.info("Empty reference handled correctly")

    except Exception as e:
        result.fail(f"Exception: {e}")

    return result


def test_numerical_tolerance_boundaries() -> TestResult:
    """Test exact boundary conditions for numerical tolerance.

    Returns:
        TestResult with validation status.
    """
    result = TestResult("Numerical Tolerance Boundaries")

    try:
        # Test exact 2% boundary
        question = "What is the value?"
        reference = "The value is 100.0"
        candidate_exact = "The value is 102.0"  # Exactly 2%

        result_json, _, _ = get_judgement_prompt.run(question, reference, candidate_exact)
        output = json.loads(result_json)

        if output["contradiction"]:
            result.fail("Should NOT have contradiction at exact 2% boundary")

        # Test absolute tolerance (1e-6)
        reference_small = "The value is 0.000001"
        candidate_small = "The value is 0.0000015"  # 5e-7 difference

        result_json, _, _ = get_judgement_prompt.run(question, reference_small, candidate_small)
        output = json.loads(result_json)

        if output["contradiction"]:
            result.fail("Should NOT have contradiction within absolute tolerance")

        # Test max of relative and absolute
        reference_mixed = "The value is 0.00001"
        candidate_mixed = "The value is 0.000011"  # 10% relative but < 1e-6 absolute

        result_json, _, _ = get_judgement_prompt.run(question, reference_mixed, candidate_mixed)
        output = json.loads(result_json)

        if output["contradiction"]:
            result.fail("Should use max(absolute, relative) tolerance")

        logger.info("Numerical tolerance boundaries verified")

    except Exception as e:
        result.fail(f"Exception: {e}")

    return result


def test_unit_conversions() -> TestResult:
    """Test automatic unit conversion handling.

    Returns:
        TestResult with validation status.
    """
    result = TestResult("Unit Conversions")

    try:
        # Length conversion
        q = "What is the distance?"
        ref = "The distance is 1 kilometer"
        cand = "The distance is 1000 meters"

        result_json, _, _ = get_judgement_prompt.run(q, ref, cand)
        output = json.loads(result_json)

        if output["contradiction"]:
            result.fail("Should convert km to meters")

        # Time conversion
        ref = "It takes 2 hours"
        cand = "It takes 120 minutes"

        result_json, _, _ = get_judgement_prompt.run(q, ref, cand)
        output = json.loads(result_json)

        if output["contradiction"]:
            result.fail("Should convert hours to minutes")

        # Currency NO conversion
        ref = "The price is 100 USD"
        cand = "The price is 90 EUR"

        result_json, _, _ = get_judgement_prompt.run(q, ref, cand)
        output = json.loads(result_json)

        if not output["contradiction"]:
            result.fail("Should NOT convert currencies")

        logger.info("Unit conversions handled correctly")

    except Exception as e:
        result.fail(f"Exception: {e}")

    return result


def test_llm_api_exceptions() -> TestResult:
    """Test handling of LLM API exceptions."""
    result = TestResult("LLM API Exception Handling")

    with patch("prompts.get_judgement_prompt.llm") as mock_llm:
        # Test 1: Network error
        print("DEBUG: Setting side_effect to ConnectionError")
        mock_llm.chat_completion.side_effect = ConnectionError("Network timeout")

        try:
            print("DEBUG: Calling run() - expecting ConnectionError")
            get_judgement_prompt.run("Q?", "Ref", "Cand")
            result.fail("Should propagate network errors")
        except ConnectionError as e:
            print(f"DEBUG: Caught expected ConnectionError: {e}")
        except Exception as e:
            print(f"DEBUG: Caught unexpected exception: {type(e).__name__}: {e}")
            result.fail(f"Wrong exception type: {type(e).__name__}: {e}")

        # Reset side_effect
        print("DEBUG: Resetting side_effect to None")
        mock_llm.chat_completion.side_effect = None

        # Test 2: Missing choices
        print("DEBUG: Setting return_value to missing choices")
        mock_llm.chat_completion.return_value = {"no_choices": "here"}

        try:
            print("DEBUG: Calling run() - expecting RuntimeError")
            get_judgement_prompt.run("Q?", "Ref", "Cand")
            result.fail("Should raise RuntimeError for missing choices")
        except RuntimeError as e:
            print(f"DEBUG: Caught RuntimeError: {e}")
            if "missing valid 'choices'" not in str(e):
                result.fail(f"Wrong error message: {e}")
        except Exception as e:
            print(f"DEBUG: Caught unexpected exception: {type(e).__name__}: {e}")
            result.fail(f"Wrong exception type: {type(e).__name__}: {e}")

    return result


def test_input_validation() -> TestResult:
    """Test validation of input parameters.

    Returns:
        TestResult with validation status.
    """
    result = TestResult("Input Parameter Validation")

    try:
        # Empty question
        try:
            get_judgement_prompt.run("", "Reference", "Candidate")
            result.fail("Should raise ValueError for empty question")
        except ValueError as e:
            if "question must be non-empty" not in str(e):
                result.fail(f"Wrong error: {e}")

        # Test string normalization
        q = "  Question\r\nwith\rdifferent\nlinebreaks  "
        ref = "  Reference\r\n  "
        cand = "  Candidate\r  "

        messages = get_judgement_prompt.get_messages(q, ref, cand)
        user_msg = messages[-1]["content"]

        # Check normalized linebreaks
        if "\r\n" in user_msg or "\r" in user_msg:
            result.fail("Should normalize all linebreaks to \\n")

        # Check trimming
        if user_msg.count("Question\nwith\ndifferent\nlinebreaks") != 1:
            result.fail("Should trim and normalize input")

        logger.info("Input validation working correctly")

    except Exception as e:
        result.fail(f"Exception: {e}")

    return result


def test_custom_params_merge() -> TestResult:
    """Test merging of custom parameters with defaults.

    Returns:
        TestResult with validation status.
    """
    result = TestResult("Custom Parameters Merge")

    try:
        with patch("prompts.get_judgement_prompt.llm") as mock_llm:
            mock_response = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "precision_c_to_r": 0.9,
                                    "recall_r_to_c": 0.9,
                                    "contradiction": False,
                                    "hallucination": False,
                                    "justification": "test",
                                    "evidence": [],
                                }
                            )
                        }
                    }
                ]
            }
            mock_llm.chat_completion.return_value = mock_response

            # Custom params should override defaults
            custom = {"temperature": 0.5, "max_tokens": 1000, "new_param": "value"}

            get_judgement_prompt.run("Q?", "Ref", "Cand", custom_params=custom)

            # Check merged params
            call_kwargs = mock_llm.chat_completion.call_args.kwargs

            if call_kwargs.get("temperature") != 0.5:
                result.fail("Custom temperature should override default")
            if call_kwargs.get("max_tokens") != 1000:
                result.fail("Custom max_tokens should override default")
            if call_kwargs.get("new_param") != "value":
                result.fail("New params should be added")

        logger.info("Custom params merged correctly")

    except Exception as e:
        result.fail(f"Exception: {e}")

    return result


def test_evidence_array_validation() -> TestResult:
    """Test evidence array validation and limits.

    Returns:
        TestResult with validation status.
    """
    result = TestResult("Evidence Array Validation")

    try:
        # Too many evidence items
        many_items = [{"source": "candidate", "quote": f"item{i}"} for i in range(10)]
        response = json.dumps(
            {
                "precision_c_to_r": 0.8,
                "recall_r_to_c": 0.8,
                "contradiction": False,
                "hallucination": False,
                "justification": "test",
                "evidence": many_items,
            }
        )

        parsed = get_judgement_prompt._parse_json_response(response)

        if len(parsed["evidence"]) > get_judgement_prompt.EVIDENCE_MAX_ITEMS:
            result.fail(f"Should limit evidence to {get_judgement_prompt.EVIDENCE_MAX_ITEMS} items")

        # Invalid source
        invalid_source = json.dumps(
            {
                "precision_c_to_r": 0.8,
                "recall_r_to_c": 0.8,
                "contradiction": False,
                "hallucination": False,
                "justification": "test",
                "evidence": [{"source": "invalid", "quote": "text"}],
            }
        )

        parsed = get_judgement_prompt._parse_json_response(invalid_source)

        for item in parsed["evidence"]:
            if item["source"] not in ["candidate", "reference"]:
                result.fail("Should only allow candidate/reference sources")

        logger.info("Evidence array validated correctly")

    except Exception as e:
        result.fail(f"Exception: {e}")

    return result


def test_f1_metric_calculation() -> TestResult:
    """Test F1 score calculation edge cases.

    Returns:
        TestResult with validation status.
    """
    result = TestResult("F1 Metric Calculation")

    try:
        # Test internal calculation: both precision and recall are 0
        metrics = get_judgement_prompt._calculate_metrics(
            {
                "precision_c_to_r": 0.0,
                "recall_r_to_c": 0.0,
                "contradiction": False,
                "hallucination": False,
            }
        )

        if metrics["f1"] != 0.0:
            result.fail("F1 should be 0 when both P and R are 0")

        # Test internal calculation: verify F1 formula
        test_cases = [
            (0.8, 0.6),  # Different values
            (1.0, 1.0),  # Perfect score
            (0.5, 0.5),  # Same values
        ]

        for p, r in test_cases:
            metrics = get_judgement_prompt._calculate_metrics(
                {
                    "precision_c_to_r": p,
                    "recall_r_to_c": r,
                    "contradiction": False,
                    "hallucination": False,
                }
            )

            expected_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

            if abs(metrics["f1"] - expected_f1) > 0.0001:
                result.fail(f"F1 calculation wrong for P={p}, R={r}")

        # Test that raw calculation works with complex numbers
        test_metrics = get_judgement_prompt._calculate_metrics(
            {
                "precision_c_to_r": 0.123456789,
                "recall_r_to_c": 0.987654321,
                "contradiction": False,
                "hallucination": False,
            }
        )

        # Just verify the formula is correct, NOT the rounding
        p = 0.123456789
        r = 0.987654321
        expected_f1_raw = 2 * p * r / (p + r)

        if abs(test_metrics["f1"] - expected_f1_raw) > 0.0001:
            result.fail("F1 calculation incorrect for complex numbers")

        # Test that rounding happens in the public API (run), not internal functions
        with patch("prompts.get_judgement_prompt.llm") as mock_llm:
            mock_response = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "precision_c_to_r": 0.123456789,
                                    "recall_r_to_c": 0.987654321,
                                    "contradiction": False,
                                    "hallucination": False,
                                    "justification": "test",
                                    "evidence": [],
                                }
                            )
                        }
                    }
                ]
            }
            mock_llm.chat_completion.return_value = mock_response

            # Test public API output
            result_json, _, _ = get_judgement_prompt.run("Q?", "Ref", "Cand")
            output = json.loads(result_json)

            # Verify F1 is rounded to 4 decimal places in public API
            f1_str = str(output["f1"])
            if "." in f1_str:
                decimal_places = len(f1_str.split(".")[1])
                if decimal_places > 4:
                    result.fail(
                        f"F1 in public API should be rounded to max 4 decimal places, got {decimal_places}"
                    )

            # Also verify the calculated value is correct
            expected_f1_rounded = round(expected_f1_raw, 4)

            if output["f1"] != expected_f1_rounded:
                result.fail(
                    f"F1 rounding incorrect: expected {expected_f1_rounded}, got {output['f1']}"
                )

        logger.info("F1 metric calculated correctly")

    except Exception as e:
        result.fail(f"Exception: {e}")

    return result


def test_score_thresholds() -> TestResult:
    """Test classification by score thresholds.

    Returns:
        TestResult with validation status.
    """
    result = TestResult("Score Threshold Classification")

    try:
        # Test exact boundaries
        test_cases = [
            (0.85, 0.85, "good"),  # Exactly at THRESHOLD_GOOD
            (0.84, 0.84, "ok"),  # Just below THRESHOLD_GOOD
            (0.70, 0.70, "ok"),  # Exactly at THRESHOLD_OK
            (0.69, 0.69, "bad"),  # Just below THRESHOLD_OK
        ]

        for p, r, expected_class in test_cases:
            metrics = get_judgement_prompt._calculate_metrics(
                {
                    "precision_c_to_r": p,
                    "recall_r_to_c": r,
                    "contradiction": False,
                    "hallucination": False,
                }
            )

            # F1 = 2PR/(P+R) = P when P=R
            # Score = F1 * 100
            expected_score = round(p * 100)

            if metrics["score"] != expected_score:
                result.fail(f"Score calculation wrong for P=R={p}")

            if metrics["class"] != expected_class:
                result.fail(f"Class should be '{expected_class}' for score {metrics['score']}")

        logger.info("Score thresholds working correctly")

    except Exception as e:
        result.fail(f"Exception: {e}")

    return result


def test_combined_penalties() -> TestResult:
    """Test combined contradiction and hallucination penalties.

    Returns:
        TestResult with validation status.
    """
    result = TestResult("Combined Penalties")

    try:
        # Both penalties
        metrics = get_judgement_prompt._calculate_metrics(
            {
                "precision_c_to_r": 0.9,
                "recall_r_to_c": 0.9,
                "contradiction": True,
                "hallucination": True,
            }
        )

        expected_penalties = (
            get_judgement_prompt.CONTRADICTION_PENALTY + get_judgement_prompt.HALLUCINATION_PENALTY
        )

        if abs(metrics["penalties"] - expected_penalties) > 0.001:
            result.fail("Penalties should sum correctly")

        # Score should never be negative
        metrics = get_judgement_prompt._calculate_metrics(
            {
                "precision_c_to_r": 0.1,
                "recall_r_to_c": 0.1,
                "contradiction": True,
                "hallucination": True,
            }
        )

        if metrics["score"] < 0:
            result.fail("Score should never be negative")

        # Verify penalty application
        f1 = 2 * 0.9 * 0.9 / (0.9 + 0.9)  # 0.9
        raw_score = f1 - expected_penalties
        expected_score = round(max(0.0, raw_score) * 100)

        metrics = get_judgement_prompt._calculate_metrics(
            {
                "precision_c_to_r": 0.9,
                "recall_r_to_c": 0.9,
                "contradiction": True,
                "hallucination": True,
            }
        )

        if metrics["score"] != expected_score:
            result.fail("Score with penalties calculated incorrectly")

        logger.info("Combined penalties applied correctly")

    except Exception as e:
        result.fail(f"Exception: {e}")

    return result


def test_multiline_answers() -> TestResult:
    """Test handling of multiline answers with different line endings.

    Returns:
        TestResult with validation status.
    """
    result = TestResult("Multiline Answer Normalization")

    try:
        q = "What is it?"
        ref = "Line 1\r\nLine 2\rLine 3\nLine 4"
        cand = "Answer\r\nwith\rmixed\nlinebreaks"

        # Format and check normalization
        prompt = get_judgement_prompt._format_user_prompt(q, ref, cand)

        if "\r\n" in prompt or "\r" in prompt.replace("\\r\\n", "").replace("\\r", ""):
            result.fail("Should normalize all linebreaks to \\n")

        # Should preserve logical structure
        if "Line 1\nLine 2\nLine 3\nLine 4" not in prompt:
            result.fail("Should preserve line structure after normalization")

        logger.info("Multiline answers normalized correctly")

    except Exception as e:
        result.fail(f"Exception: {e}")

    return result


def test_long_answers() -> TestResult:
    """Test handling of extremely long answers.

    Returns:
        TestResult with validation status.
    """
    result = TestResult("Long Answer Handling")

    try:
        # Create very long justification
        long_just = " ".join(["word"] * 100)  # 100 words

        response = json.dumps(
            {
                "precision_c_to_r": 0.8,
                "recall_r_to_c": 0.8,
                "contradiction": False,
                "hallucination": False,
                "justification": long_just,
                "evidence": [],
            }
        )

        parsed = get_judgement_prompt._parse_json_response(response)

        # Justification should be preserved (max is checked in prompt)
        word_count = len(parsed["justification"].split())

        # Module doesn't truncate, relies on LLM following instructions
        if not parsed["justification"]:
            result.fail("Should preserve justification")

        # Test near max_tokens limit
        with patch("prompts.get_judgement_prompt.llm") as mock_llm:
            # Create response near 800 token limit
            big_response = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "precision_c_to_r": 0.8,
                                    "recall_r_to_c": 0.8,
                                    "contradiction": False,
                                    "hallucination": False,
                                    "justification": "x" * 400,  # Long but valid
                                    "evidence": [],
                                }
                            )
                        }
                    }
                ]
            }
            mock_llm.chat_completion.return_value = big_response

            # Should handle without error
            result_json, _, _ = get_judgement_prompt.run("Q?", "Ref", "Cand")
            output = json.loads(result_json)

            if not output:
                result.fail("Should handle long responses")

        logger.info("Long answers handled correctly")

    except Exception as e:
        result.fail(f"Exception: {e}")

    return result


def run_all_tests() -> tuple[int, int]:
    """Execute all test cases and report results.

    Returns:
        Tuple of (passed_count, failed_count).
    """
    # Define test suite - original tests plus new ones
    tests = [
        test_initialization,
        test_perfect_match,
        test_partial_match,
        test_contradiction_detection,
        test_hallucination_detection,
        test_empty_candidate,
        test_numerical_tolerance,
        # New tests
        test_invalid_json_response,
        test_incomplete_llm_response,
        test_empty_reference,
        test_numerical_tolerance_boundaries,
        test_unit_conversions,
        test_llm_api_exceptions,
        test_input_validation,
        test_custom_params_merge,
        test_evidence_array_validation,
        test_f1_metric_calculation,
        test_score_thresholds,
        test_combined_penalties,
        test_multiline_answers,
        test_long_answers,
    ]

    results: list[TestResult] = []

    print("=== Symmetric Entailment Judgement Test Suite ===\n")

    # Execute each test
    for i, test_func in enumerate(tests, 1):
        print(f"Test {i}: {test_func.__name__.replace('test_', '').replace('_', ' ').title()}")

        result = test_func()
        results.append(result)

        if result.passed:
            print(f"✓ PASSED: {result.name}\n")
        else:
            print(f"✗ FAILED: {result.name}")
            for error in result.errors:
                print(f"  - {error}")
            print()

    # Calculate summary
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    return passed, failed


def main() -> None:
    """Main test execution entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s][%(filename)s:%(lineno)d][%(message)s]",
    )

    passed, failed = run_all_tests()

    # Print summary
    print("=== Test Summary ===")
    print(f"Tests run: {passed + failed}")
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")

    if failed == 0:
        print("\n✓ ALL TESTS PASSED")
        exit(0)
    else:
        print(f"\n✗ {failed} TEST(S) FAILED - Review output above for details")
        print("\nNote: Some failures may be due to LLM non-determinism despite temperature=0.0")
        exit(1)


if __name__ == "__main__":
    main()
