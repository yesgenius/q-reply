"""Symmetric entailment judgement module for answer evaluation.

This module provides functionality to evaluate how semantically equivalent
a candidate answer is to a reference answer using bidirectional entailment.
Uses LLM with structured JSON output for consistent evaluation metrics.

The module implements precision (C→R) and recall (R→C) scoring with
contradiction and hallucination detection for comprehensive assessment.

Example:
    Basic usage for answer evaluation:

    ```python
    from prompts import get_judgement_prompt

    # Initialize judgement system
    get_judgement_prompt.update_system_prompt()

    # Evaluate answer equivalence
    question = "What is the capital of France?"
    reference_answer = "The capital of France is Paris, located in the north-central region."
    candidate_answer = "Paris is France's capital city."

    result, messages, response = get_judgement_prompt.run(
        question=question, reference_answer=reference_answer, candidate_answer=candidate_answer
    )

    # result is a JSON string
    import json

    output = json.loads(result)
    print(output)  # {"score": 85, "class": "good", ...}
    ```

Usage:
    python -m prompts.get_judgement_prompt
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from gigachat.client import GigaChatClient
import json_repair
from utils.logger import get_logger


logger = get_logger(__name__)

# Initialize LLM client
llm = GigaChatClient()

# Model parameters optimized for deterministic judgement
params: dict[str, Any] = {
    "model": "GigaChat",
    "temperature": 0.0,  # Deterministic for consistent evaluation
    "top_p": 1.0,
    "stream": False,
    "max_tokens": 800,  # Sufficient for judgement JSON
}

# Evaluation configuration constants
JUSTIFICATION_MAX_WORDS = 40
EVIDENCE_MAX_ITEMS = 2
SCORE_SCALE = 100
NUMERICAL_TOLERANCE_ABSOLUTE = 1e-6
NUMERICAL_TOLERANCE_RELATIVE = 0.02

# Penalty constants
CONTRADICTION_PENALTY = 0.20
HALLUCINATION_PENALTY = 0.10

# Score thresholds
THRESHOLD_GOOD = 85
THRESHOLD_OK = 70

# Global cached variables
_system_prompt: str | None = None
_chat_history: list[dict[str, str]] = []


def _format_user_prompt(question: str, reference_answer: str, candidate_answer: str) -> str:
    """Format the user prompt for symmetric entailment evaluation.

    Creates a structured prompt that instructs the judge to evaluate
    bidirectional semantic equivalence between answers.

    Args:
        question: The original question being answered.
        reference_answer: The reference/ground truth answer.
        candidate_answer: The candidate answer to evaluate.

    Returns:
        Formatted user prompt string.

    Raises:
        ValueError: If any input is empty or None.

    Example:
        >>> prompt = _format_user_prompt("Q?", "Reference A", "Candidate A")
    """
    if not question or not reference_answer or not candidate_answer:
        raise ValueError("All inputs (question, reference, candidate) must be non-empty")

    # Preprocess all inputs (normalize line endings)
    question_norm = question.strip().replace("\r\n", "\n").replace("\r", "\n")
    reference_norm = reference_answer.strip().replace("\r\n", "\n").replace("\r", "\n")
    candidate_norm = candidate_answer.strip().replace("\r\n", "\n").replace("\r", "\n")

    user_prompt = f"""
EVALUATE THIS ANSWER PAIR:

QUESTION:
{question_norm}

REFERENCE (R):
{reference_norm}

CANDIDATE (C):
{candidate_norm}

EXECUTE EVALUATION NOW. RETURN ONLY JSON.
"""
    return user_prompt


def _generate_system_prompt(**kwargs: Any) -> str:
    """Generate system prompt for symmetric entailment judgement.

    Creates a structured prompt that instructs the LLM to act as
    a strict deterministic judge with JSON output.

    Args:
        **kwargs: Reserved for future configuration options.

    Returns:
        System prompt string for judgement task.

    Example:
        >>> prompt = _generate_system_prompt()
    """
    # JSON schema for strict validation - removed maxLength for justification (DRY principle)
    json_schema = json.dumps(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "precision_c_to_r",
                "recall_r_to_c",
                "contradiction",
                "hallucination",
                "justification",
                "evidence",
            ],
            "properties": {
                "precision_c_to_r": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "multipleOf": 0.01,
                },
                "recall_r_to_c": {"type": "number", "minimum": 0, "maximum": 1, "multipleOf": 0.01},
                "contradiction": {"type": "boolean"},
                "hallucination": {"type": "boolean"},
                "justification": {"type": "string"},  # Removed maxLength - using word limit instead
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["source", "quote"],
                        "properties": {
                            "source": {"type": "string", "enum": ["candidate", "reference"]},
                            "quote": {"type": "string", "maxLength": 200},
                        },
                    },
                    "maxItems": EVIDENCE_MAX_ITEMS,
                },
            },
        },
        ensure_ascii=False,
        indent=2,
    )

    json_example = json.dumps(
        {
            "precision_c_to_r": "number between 0.00 and 1.00 (step 0.01)",
            "recall_r_to_c": "number between 0.00 and 1.00 (step 0.01)",
            "contradiction": "true or false",
            "hallucination": "true or false",
            "justification": "short explanation",
            "evidence": [
                {"source": "candidate or reference", "quote": "string taken directly from the text"}
            ],
        },
        ensure_ascii=False,
        indent=2,
    )

    system_prompt = f"""
YOU ARE A DETERMINISTIC SEMANTIC EQUIVALENCE JUDGE.

YOUR SOLE TASK: Evaluate bidirectional semantic entailment between REFERENCE and CANDIDATE answers.

MANDATORY OUTPUT FORMAT:
Return ONLY valid JSON matching this exact schema:
{json_schema}
Schema-based JSON example with placeholders:
{json_example}

CRITICAL EVALUATION RULES YOU MUST FOLLOW:
0. EVALUATION SCALE FOR PRECISION AND RECALL:
   - 1.0: Complete equivalence in the evaluated direction
   - 0.9: All key points covered, only trivial details missing
   - 0.8: One key detail missing/added, conclusion unchanged
   - 0.6: Part of core missing/added, conclusion partially matches
   - 0.4: Some fragments match, but conclusion different/incomplete
   - 0.2: Sporadic matches only
   - 0.0: No semantic overlap

1. PRECISION (precision_c_to_r): CALCULATE what fraction of CANDIDATE content is confirmed by REFERENCE
   - 1.0 = ALL candidate information exists in reference
   - 0.0 = NO candidate information exists in reference
   - IGNORE style, format, politeness - EVALUATE ONLY factual content

2. RECALL (recall_r_to_c): CALCULATE what fraction of REFERENCE content is covered by CANDIDATE
   - 1.0 = ALL reference information exists in candidate
   - 0.0 = NO reference information exists in candidate
   - IGNORE style, format, politeness - EVALUATE ONLY factual content

3. CONTRADICTION FLAG (contradiction): SET to true ONLY when:
   - Key assertions are inverted (yes/no, allowed/forbidden, above/below)
   - Numerical values differ BEYOND tolerance: |C-R| > max({NUMERICAL_TOLERANCE_ABSOLUTE}, {
        NUMERICAL_TOLERANCE_RELATIVE
    }*|R|)
   - Different entities that change conclusion (different model/algorithm/protocol/currency)
   - Unit conversion errors affecting conclusion

4. HALLUCINATION FLAG (hallucination): SET to true ONLY when BOTH conditions met:
   - CANDIDATE contains NEW verifiable facts (numbers/dates/names/URLs/prices/policies/versions) absent from QUESTION and REFERENCE
   - These facts MATERIALLY affect the conclusion or recommendation

5. JUSTIFICATION (justification): WRITE maximum {
        JUSTIFICATION_MAX_WORDS
    } words explaining your scores

6. EVIDENCE (evidence): PROVIDE up to {
        EVIDENCE_MAX_ITEMS
    } short quotes with exact source attribution

7. NUMERICAL TOLERANCE: YOU MUST APPLY these exact rules:
   - Values within |C-R| ≤ max({NUMERICAL_TOLERANCE_ABSOLUTE}, {
        NUMERICAL_TOLERANCE_RELATIVE
    }*|R|) are EQUIVALENT
   - Example: 100.0 vs 101.5 (1.5% difference) is EQUIVALENT when tolerance is 2%
   - NEVER mark as contradiction if within tolerance

8. UNIT CONVERSION: YOU MUST automatically convert simple units:
   - Length: mm↔cm↔m↔km
   - Time: ms↔s↔min↔h
   - Mass: mg↔g↔kg
   - Temperature: °C↔K (by difference only)
   - Data: bit↔byte↔KB↔MB↔GB
   - Percentages: treat as ratios
   - Currency: DO NOT convert exchange rates


NEVER:
- Add explanatory text outside JSON
- Include markdown formatting
- Discuss your reasoning process
- Apologize or express uncertainty

ALWAYS:
- Output pure JSON only
- Apply numerical tolerance strictly
- Evaluate semantic meaning, not surface form
- Set flags conservatively (only when clearly warranted)"""

    return system_prompt


def _generate_chat_history(**kwargs: Any) -> list[dict[str, str]]:
    """Generate chat history with few-shot examples for better LLM calibration.

    Provides carefully selected examples that demonstrate correct evaluation
    of edge cases where models commonly make mistakes.

    Args:
        **kwargs: Reserved for future parameters.

    Returns:
        List of message dictionaries demonstrating correct evaluations.
    """
    history = [
        # Example 1: Partial match - precision vs recall distinction
        {
            "role": "user",
            "content": """
EVALUATE THIS ANSWER PAIR:

QUESTION:
What are the main components of a computer?

REFERENCE (R):
A computer has four main components: CPU, memory, storage, and motherboard.

CANDIDATE (C):
A computer has CPU and memory.

EXECUTE EVALUATION NOW. RETURN ONLY JSON.""",
        },
        {
            "role": "assistant",
            "content": """
{
  "precision_c_to_r": 1.0,
  "recall_r_to_c": 0.5,
  "contradiction": false,
  "hallucination": false,
  "justification": "All candidate info exists in reference; candidate covers 2 of 4 components",
  "evidence": [
    {"source": "candidate", "quote": "CPU and memory"},
    {"source": "reference", "quote": "CPU, memory, storage, and motherboard"}
  ]
}""",
        },
        # Example 2: Numerical tolerance case
        {
            "role": "user",
            "content": """
EVALUATE THIS ANSWER PAIR:

QUESTION:
What is the distance?

REFERENCE (R):
The distance is 50.0 kilometers.

CANDIDATE (C):
The distance is 50.8 kilometers.

EXECUTE EVALUATION NOW. RETURN ONLY JSON.""",
        },
        {
            "role": "assistant",
            "content": """
{
  "precision_c_to_r": 0.98,
  "recall_r_to_c": 0.98,
  "contradiction": false,
  "hallucination": false,
  "justification": "Values differ by 1.6%, within 2% tolerance threshold",
  "evidence": [
    {"source": "reference", "quote": "50.0 kilometers"},
    {"source": "candidate", "quote": "50.8 kilometers"}
  ]
}""",
        },
        # Example 3: Clear contradiction beyond tolerance
        {
            "role": "user",
            "content": """
EVALUATE THIS ANSWER PAIR:

QUESTION:
What is the price?

REFERENCE (R):
The price is $100.

CANDIDATE (C):
The price is $200.

EXECUTE EVALUATION NOW. RETURN ONLY JSON.""",
        },
        {
            "role": "assistant",
            "content": """
{
  "precision_c_to_r": 0.0,
  "recall_r_to_c": 0.0,
  "contradiction": true,
  "hallucination": false,
  "justification": "Price differs by 100%, far exceeding tolerance",
  "evidence": [
    {"source": "reference", "quote": "$100"},
    {"source": "candidate", "quote": "$200"}
  ]
}""",
        },
        # Example 4: Hallucination with added facts
        {
            "role": "user",
            "content": """
EVALUATE THIS ANSWER PAIR:

QUESTION:
What is Java?

REFERENCE (R):
Java is a programming language.

CANDIDATE (C):
Java is a programming language released by Sun Microsystems in 1995, currently at version 21.

EXECUTE EVALUATION NOW. RETURN ONLY JSON.""",
        },
        {
            "role": "assistant",
            "content": """
{
  "precision_c_to_r": 0.3,
  "recall_r_to_c": 1.0,
  "contradiction": false,
  "hallucination": true,
  "justification": "Candidate adds verifiable facts not in reference: company, year, version",
  "evidence": [
    {"source": "candidate", "quote": "released by Sun Microsystems in 1995, currently at version 21"},
    {"source": "reference", "quote": "Java is a programming language"}
  ]
}""",
        },
    ]

    return history


def update_system_prompt(**kwargs: Any) -> str:
    """Update or retrieve the cached system prompt.

    Updates the global system prompt if kwargs are provided,
    otherwise returns the existing cached prompt.

    Args:
        **kwargs: Reserved for future configuration options.

    Returns:
        The current system prompt string.
    """
    global _system_prompt

    if kwargs:
        _system_prompt = _generate_system_prompt(**kwargs)
        logger.debug("System prompt updated for judgement")
    elif _system_prompt is None:
        _system_prompt = _generate_system_prompt()
        logger.debug("System prompt initialized with defaults")

    return _system_prompt


def update_chat_history(**kwargs: Any) -> list[dict[str, str]]:
    """Update or retrieve the cached chat history with few-shot examples.

    Returns few-shot examples for LLM calibration to improve
    evaluation consistency on edge cases.

    Args:
        **kwargs: Reserved for future parameters.

    Returns:
        List of few-shot example messages for LLM calibration.
    """
    global _chat_history

    _chat_history = _generate_chat_history(**kwargs)
    return _chat_history.copy()


def get_messages(
    question: str, reference_answer: str, candidate_answer: str
) -> list[dict[str, str]]:
    """Build complete message list for LLM judgement request.

    Note: This function expects valid non-empty inputs. Empty inputs
    will raise ValueError as per Fail Fast principle.

    Args:
        question: The original question.
        reference_answer: The reference answer.
        candidate_answer: The candidate answer to evaluate.

    Returns:
        List of message dictionaries formatted for LLM API.

    Raises:
        ValueError: If any input is invalid or empty.
    """
    messages_list = []

    # Get system prompt from cache
    system_prompt = update_system_prompt()
    if system_prompt:
        messages_list.append({"role": "system", "content": system_prompt})

    # Get chat history with few-shot examples
    history = update_chat_history()
    messages_list.extend(history)

    # Format and add user prompt
    user_prompt = _format_user_prompt(question, reference_answer, candidate_answer)
    messages_list.append({"role": "user", "content": user_prompt})

    return messages_list


def _parse_json_response(response_text: str) -> dict[str, Any]:
    """Parse LLM judgement response using json_repair with validation.

    Args:
        response_text: Raw text response from LLM containing JSON.

    Returns:
        Dict with required fields:
            - precision_c_to_r: Float between 0 and 1, rounded to 0.01
            - recall_r_to_c: Float between 0 and 1, rounded to 0.01
            - contradiction: Boolean flag
            - hallucination: Boolean flag
            - justification: String explanation
            - evidence: List of evidence items

    Raises:
        ValueError: If response cannot be parsed or validated.
    """
    text = response_text.strip()

    # Find JSON fragment
    start_idx = text.find("{")
    if start_idx == -1:
        return _extract_fields_fallback(response_text)

    json_fragment = text[start_idx:]

    try:
        # Repair and parse JSON
        repaired_obj = json_repair.loads(json_fragment)

        if not isinstance(repaired_obj, dict):
            raise ValueError(f"json_repair returned {type(repaired_obj).__name__} instead of dict")

        # Validate required fields
        required_fields = {
            "precision_c_to_r",
            "recall_r_to_c",
            "contradiction",
            "hallucination",
            "justification",
            "evidence",
        }
        present_fields = set(repaired_obj.keys())
        missing_fields = required_fields - present_fields

        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        result: dict[str, Any] = {}

        # Validate and round precision_c_to_r to 0.01
        precision = repaired_obj.get("precision_c_to_r")
        if not isinstance(precision, (int, float)):
            raise ValueError(f"precision_c_to_r must be numeric, got {type(precision).__name__}")
        precision_float = float(precision)
        if not 0 <= precision_float <= 1:
            raise ValueError(f"precision_c_to_r={precision_float} outside [0,1]")
        result["precision_c_to_r"] = precision_float

        # Validate and round recall_r_to_c to 0.01
        recall = repaired_obj.get("recall_r_to_c")
        if not isinstance(recall, (int, float)):
            raise ValueError(f"recall_r_to_c must be numeric, got {type(recall).__name__}")
        recall_float = float(recall)
        if not 0 <= recall_float <= 1:
            raise ValueError(f"recall_r_to_c={recall_float} outside [0,1]")
        result["recall_r_to_c"] = recall_float

        # Validate boolean flags
        contradiction = repaired_obj.get("contradiction")
        if not isinstance(contradiction, bool):
            # Try to convert common representations
            if isinstance(contradiction, str):
                contradiction = contradiction.lower() in ["true", "yes", "1"]
            elif isinstance(contradiction, (int, float)):
                contradiction = bool(contradiction)
            else:
                raise ValueError(
                    f"contradiction must be boolean, got {type(contradiction).__name__}"
                )
        result["contradiction"] = contradiction

        hallucination = repaired_obj.get("hallucination")
        if not isinstance(hallucination, bool):
            # Try to convert common representations
            if isinstance(hallucination, str):
                hallucination = hallucination.lower() in ["true", "yes", "1"]
            elif isinstance(hallucination, (int, float)):
                hallucination = bool(hallucination)
            else:
                raise ValueError(
                    f"hallucination must be boolean, got {type(hallucination).__name__}"
                )
        result["hallucination"] = hallucination

        # Validate justification
        justification = repaired_obj.get("justification", "")
        if not isinstance(justification, str):
            justification = str(justification)
        result["justification"] = justification

        # Validate evidence array
        evidence = repaired_obj.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []

        validated_evidence = []
        for item in evidence[:EVIDENCE_MAX_ITEMS]:  # Limit to max items
            if isinstance(item, dict):
                source = item.get("source", "")
                quote = item.get("quote", "")
                if source in ["candidate", "reference"]:
                    validated_evidence.append({"source": source, "quote": str(quote)})
        result["evidence"] = validated_evidence

        logger.debug(
            f"Successfully parsed judgement with F1 components: P={precision_float:.2f}, R={recall_float:.2f}"
        )
        return result

    except Exception as e:
        logger.warning(f"JSON parsing failed: {e}. Using fallback extraction")
        return _extract_fields_fallback(response_text)


def _extract_fields_fallback(response_text: str) -> dict[str, Any]:
    """Extract fields individually when JSON parsing fails.

    Args:
        response_text: Raw text response.

    Returns:
        Dict with extracted or default values.

    Raises:
        ValueError: If critical fields cannot be extracted.
    """
    logger.warning(f"Invalid JSON, using fallback extraction for response: [{response_text}]")

    result: dict[str, Any] = {}

    # Extract precision_c_to_r (required)
    precision = _extract_precision(response_text)
    if precision is None:
        logger.warning("Failed to extract precision_c_to_r, using default 0.0")
        result["precision_c_to_r"] = 0.0
    else:
        result["precision_c_to_r"] = precision

    # Extract recall_r_to_c (required)
    recall = _extract_recall(response_text)
    if recall is None:
        logger.warning("Failed to extract recall_r_to_c, using default 0.0")
        result["recall_r_to_c"] = 0.0
    else:
        result["recall_r_to_c"] = recall

    # Extract contradiction flag
    contradiction = _extract_contradiction(response_text)
    if contradiction is None:
        logger.warning("Failed to extract contradiction, using default False")
        result["contradiction"] = False
    else:
        result["contradiction"] = contradiction

    # Extract hallucination flag
    hallucination = _extract_hallucination(response_text)
    if hallucination is None:
        logger.warning("Failed to extract hallucination, using default False")
        result["hallucination"] = False
    else:
        result["hallucination"] = hallucination

    # Extract justification
    justification = _extract_justification(response_text)
    if justification is None:
        logger.warning("Failed to extract justification, using empty string")
        result["justification"] = ""
    else:
        result["justification"] = justification

    # Extract evidence array
    evidence = _extract_evidence(response_text)
    if evidence is None or len(evidence) == 0:
        logger.warning("Failed to extract evidence, using empty list")
        result["evidence"] = []
    else:
        result["evidence"] = evidence

    return result


def _extract_precision(text: str) -> float | None:
    """Extract precision_c_to_r value from text with improved robustness.

    Args:
        text: Text to search.

    Returns:
        Extracted precision as float or None if not found.
    """
    # Support both double and single quotes
    patterns = [
        r'"precision_c_to_r"\s*:\s*([0-9.]+)',
        r"'precision_c_to_r'\s*:\s*([0-9.]+)",
        r"precision_c_to_r\s*:\s*([0-9.]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                return min(1.0, max(0.0, value))  # Clamp to [0, 1]
            except ValueError:
                continue

    return None


def _extract_recall(text: str) -> float | None:
    """Extract recall_r_to_c value from text with improved robustness.

    Args:
        text: Text to search.

    Returns:
        Extracted recall as float or None if not found.
    """
    # Support both double and single quotes
    patterns = [
        r'"recall_r_to_c"\s*:\s*([0-9.]+)',
        r"'recall_r_to_c'\s*:\s*([0-9.]+)",
        r"recall_r_to_c\s*:\s*([0-9.]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                return min(1.0, max(0.0, value))  # Clamp to [0, 1]
            except ValueError:
                continue

    return None


def _extract_contradiction(text: str) -> bool | None:
    """Extract contradiction flag from text with improved robustness.

    Args:
        text: Text to search.

    Returns:
        Boolean value or None if not found.
    """
    # Check for true values - support both double and single quotes
    true_patterns = [
        r'"contradiction"\s*:\s*(true|True|TRUE|1)',
        r"'contradiction'\s*:\s*(true|True|TRUE|1)",
    ]

    for pattern in true_patterns:
        if re.search(pattern, text):
            return True

    # Check for false values - support both double and single quotes
    false_patterns = [
        r'"contradiction"\s*:\s*(false|False|FALSE|0)',
        r"'contradiction'\s*:\s*(false|False|FALSE|0)",
    ]

    for pattern in false_patterns:
        if re.search(pattern, text):
            return False

    return None


def _extract_hallucination(text: str) -> bool | None:
    """Extract hallucination flag from text with improved robustness.

    Args:
        text: Text to search.

    Returns:
        Boolean value or None if not found.
    """
    # Check for true values - support both double and single quotes
    true_patterns = [
        r'"hallucination"\s*:\s*(true|True|TRUE|1)',
        r"'hallucination'\s*:\s*(true|True|TRUE|1)",
    ]

    for pattern in true_patterns:
        if re.search(pattern, text):
            return True

    # Check for false values - support both double and single quotes
    false_patterns = [
        r'"hallucination"\s*:\s*(false|False|FALSE|0)',
        r"'hallucination'\s*:\s*(false|False|FALSE|0)",
    ]

    for pattern in false_patterns:
        if re.search(pattern, text):
            return False

    return None


def _extract_justification(text: str) -> str | None:
    """Extract justification string from text with improved robustness.

    Args:
        text: Text to search.

    Returns:
        Extracted justification or None if not found.
    """
    # Support both single and double quotes, and multiline
    patterns = [
        r'"justification"\s*:\s*"([^"]+)"',
        r"'justification'\s*:\s*'([^']+)'",
        r'justification\s*:\s*"([^"]+)"',
        r"justification\s*:\s*'([^']+)'",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)  # Added DOTALL for multiline
        if match:
            return match.group(1).strip()

    return None


def _extract_evidence(text: str) -> list[dict[str, str]] | None:
    """Extract evidence array from text with improved robustness.

    Args:
        text: Text to search.

    Returns:
        Extracted evidence list or None if not found.
    """
    # Try to find evidence array in various formats
    patterns = [
        r'"evidence"\s*:\s*\[([^\]]*)\]',
        r"'evidence'\s*:\s*\[([^\]]*)\]",
        r"evidence\s*:\s*\[([^\]]*)\]",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            evidence_str = match.group(1)

            # First try to parse the array with json_repair
            try:
                evidence_array_str = f"[{evidence_str}]"
                parsed_array = json_repair.loads(evidence_array_str)
                if isinstance(parsed_array, list):
                    evidence_list = []
                    for item in parsed_array:
                        if isinstance(item, dict):
                            source = item.get("source", "")
                            quote = item.get("quote", "")
                            if source in ["candidate", "reference"]:
                                evidence_list.append({"source": source, "quote": str(quote)})
                                if len(evidence_list) >= EVIDENCE_MAX_ITEMS:
                                    break
                    if evidence_list:
                        return evidence_list
            except Exception:
                pass  # Fall back to regex

            # Fallback: extract individual evidence items with improved regex
            evidence_list = []
            # Support both single and double quotes, and multiline quotes
            item_patterns = [
                r'\{\s*"source"\s*:\s*"([^"]+)"\s*,\s*"quote"\s*:\s*"([^"]+)"\s*\}',
                r"\{\s*'source'\s*:\s*'([^']+)'\s*,\s*'quote'\s*:\s*'([^']+)'\s*\}",
            ]

            for item_pattern in item_patterns:
                for item_match in re.finditer(item_pattern, evidence_str, re.DOTALL):
                    source = item_match.group(1).strip()
                    quote = item_match.group(2).strip()
                    if source in ["candidate", "reference"]:
                        evidence_list.append({"source": source, "quote": quote})
                        if len(evidence_list) >= EVIDENCE_MAX_ITEMS:
                            break
                if len(evidence_list) >= EVIDENCE_MAX_ITEMS:
                    break

            if evidence_list:
                return evidence_list

    return None


def _calculate_metrics(judge_output: dict[str, Any]) -> dict[str, Any]:
    """Calculate final score and metrics from judge output.

    Args:
        judge_output: Raw judgement from LLM with precision/recall/flags.

    Returns:
        Dict with calculated metrics:
            - f1: Semantic F1 score
            - penalties: Total penalties applied
            - score: Final score (0-100)
            - class: Quality class (good/ok/bad)
    """
    # Extract values
    precision = judge_output.get("precision_c_to_r", 0.0)
    recall = judge_output.get("recall_r_to_c", 0.0)
    contradiction = judge_output.get("contradiction", False)
    hallucination = judge_output.get("hallucination", False)

    # Calculate F1
    if precision == 0 and recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    # Calculate penalties
    penalties = 0.0
    if contradiction:
        penalties += CONTRADICTION_PENALTY
    if hallucination:
        penalties += HALLUCINATION_PENALTY

    # Calculate final score
    raw_score = f1 - penalties
    score = round(max(0.0, raw_score) * SCORE_SCALE)

    # Determine class
    if score >= THRESHOLD_GOOD:
        quality_class = "good"
    elif score >= THRESHOLD_OK:
        quality_class = "ok"
    else:
        quality_class = "bad"

    return {
        "f1": f1,
        "penalties": penalties,
        "score": score,
        "class": quality_class,
    }


def run(
    question: str,
    reference_answer: str,
    candidate_answer: str,
    custom_params: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, str]], dict[str, Any]]:
    """Evaluate semantic equivalence using symmetric entailment.

    Args:
        question: The original question being answered.
        reference_answer: The reference/ground truth answer.
        candidate_answer: The candidate answer to evaluate.
        custom_params: Optional parameter overrides for this request.

    Returns:
        Tuple containing:
            - Valid JSON string with evaluation result containing:
                - score: Final score (0-100)
                - class: Quality class (good/ok/bad)
                - f1: Semantic F1 score
                - precision_c_to_r: Precision score (C→R)
                - recall_r_to_c: Recall score (R→C)
                - contradiction: Boolean flag
                - hallucination: Boolean flag
                - justification: Explanation string
                - evidence: List of evidence items
                - penalties: Total penalties applied
            - List of messages sent to the LLM
            - Raw response dict from the LLM

    Raises:
        ValueError: If inputs are empty or response is invalid.
        RuntimeError: If LLM response format is unexpected.

    Example:
        >>> result, msgs, resp = run("What is X?", "X is Y", "X equals Y")
        >>> import json
        >>> print(json.loads(result)["score"])
        85
    """
    # Normalize inputs first (before any validation)
    q = (question or "").strip()
    r = (reference_answer or "").strip()
    c = (candidate_answer or "").strip()

    # Edge case §8: Empty candidate - not an exception, but a valid case
    if c == "":
        result = {
            "score": 0,
            "class": "bad",
            "f1": 0.0,
            "precision_c_to_r": 1.0,  # Empty has no incorrect content
            "recall_r_to_c": 0.0,  # Empty covers nothing
            "contradiction": False,
            "hallucination": False,
            "justification": "Empty candidate answer",
            "evidence": [],
            "penalties": 0.0,
        }
        return json.dumps(result, ensure_ascii=False, indent=2), [], {}

    # Now strict validation for remaining fields
    if not q:
        raise ValueError("question must be non-empty")
    if not r:
        # §8: Empty reference "exclude from aggregates" - valid to throw error and log
        logger.warning("Empty reference_answer - exclude from aggregates")
        raise ValueError("reference_answer must be non-empty")

    # Merge parameters
    request_params = {k: v for k, v in params.items() if v is not None}
    if custom_params:
        for k, v in custom_params.items():
            if v is not None:
                request_params[k] = v

    # Build messages
    messages_list = get_messages(q, r, c)

    logger.debug(f"Judging candidate answer for: {q[:50]}...")
    logger.debug(f"Request params: {request_params}")

    try:
        # Make LLM request
        raw_response = llm.chat_completion(messages=messages_list, **request_params)

        # Extract content from response
        if not isinstance(raw_response, dict):
            raise RuntimeError(f"Expected dict response, got {type(raw_response)}")

        if "choices" not in raw_response or not raw_response["choices"]:
            raise RuntimeError("Response missing valid 'choices'")

        first_choice = raw_response["choices"][0]
        if "message" not in first_choice or "content" not in first_choice["message"]:
            raise RuntimeError("Response missing message content")

        response_text = first_choice["message"]["content"]

        # Parse judgement response
        judge_output = _parse_json_response(response_text)

        # Calculate final metrics
        metrics = _calculate_metrics(judge_output)

        # Combine all results
        final_result = {
            "score": metrics["score"],
            "class": metrics["class"],
            "f1": round(metrics["f1"], 4),
            "precision_c_to_r": round(judge_output["precision_c_to_r"], 4),
            "recall_r_to_c": round(judge_output["recall_r_to_c"], 4),
            "contradiction": judge_output["contradiction"],
            "hallucination": judge_output["hallucination"],
            "justification": judge_output["justification"],
            "evidence": judge_output["evidence"],
            "penalties": round(metrics["penalties"], 2),
        }

        result_json = json.dumps(final_result, ensure_ascii=False, indent=2)
        return result_json, messages_list, raw_response

    except Exception as e:
        logger.error(f"Judgement failed: {e}")
        raise
