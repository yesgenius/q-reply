"""Answer generation prompt module for RAG system.

This module provides functionality to generate answers to conference questions
using context from previous Q&A pairs and optional conference topic.
Uses LLM with structured JSON output for consistent response format.

The module caches system prompts to avoid redundant computations
while maintaining flexibility for dynamic context.

Example:
    Basic usage with conference context:

    ```python
    from prompts import get_answer_prompt

    # Define conference topic (optional)
    topic = "Machine Learning in Production"

    # Previous Q&A pairs for context
    qa_pairs = [
        {
            "question": "How do you handle model versioning?",
            "answer": "We use MLflow for tracking models and DVC for data versioning."
        },
        {
            "question": "What's your approach to A/B testing?",
            "answer": "We implement gradual rollouts with feature flags and statistical analysis."
        }
    ]

    # Initialize answer generation system
    get_answer_prompt.update_system_prompt(topic=topic)

    # Generate answer using context
    question = "How do you monitor model performance?"
    result, messages = get_answer_prompt.run(
        user_question=question,
        qa_pairs=qa_pairs
    )
    print(result)  # {"answer": "Based on context, monitoring can include..."}
    ```

Usage:
    python -m prompts.get_answer_prompt
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from gigachat.client import GigaChatClient

logger = logging.getLogger(__name__)

# Initialize LLM client
llm = GigaChatClient()

# Model parameters optimized for answer generation
params: Dict[str, Any] = {
    "model": "GigaChat",
    "temperature": 0.3,  # Balanced for informative yet creative answers
    "top_p": 0.95,
    "stream": False,
    "max_tokens": 1000,  # Sufficient for comprehensive answers
    "repetition_penalty": 1.1,
}

# Global cached variables
_system_prompt: Optional[str] = None
_chat_history: List[Dict[str, str]] = []
_current_topic: Optional[str] = None  # Cache current topic for validation


def _format_user_prompt(question: str, qa_pairs: List[Dict[str, str]]) -> str:
    """Format the user prompt with question and context Q&A pairs.

    Creates a structured user prompt that includes the question
    and relevant Q&A pairs from previous conferences for context.

    Args:
        question: The user's question to answer.
        qa_pairs: List of dictionaries with 'question' and 'answer' keys
            providing context from previous conferences.

    Returns:
        Formatted user prompt string.

    Raises:
        ValueError: If qa_pairs is empty or invalid format.

    Example:
        >>> qa_pairs = [
        ...     {"question": "What is Docker?", "answer": "Container platform..."}
        ... ]
        >>> prompt = _format_user_prompt("How to scale?", qa_pairs)
    """
    if not qa_pairs:
        raise ValueError("qa_pairs cannot be empty - context is required")

    # Validate qa_pairs structure
    for i, pair in enumerate(qa_pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"qa_pairs[{i}] must be a dictionary")
        if "question" not in pair or "answer" not in pair:
            raise ValueError(f"qa_pairs[{i}] must have 'question' and 'answer' keys")

    # Format context Q&A pairs
    context_parts = []
    for i, pair in enumerate(qa_pairs, 1):
        context_parts.append(
            f"Context {i}:\n"
            f"Question: {pair['question']}\n"
            f"Answer: {pair['answer']}"
        )

    context_text = "\n\n".join(context_parts)

    # Create user prompt with context and question
    user_prompt = (
        f"Here are relevant Q&A pairs from previous conferences that may help:\n\n"
        f"{context_text}\n\n"
        f"---\n\n"
        f"Current question: {question}\n\n"
        f"Based on the context above and your knowledge, provide a comprehensive answer to the current question."
    )

    return user_prompt


def _generate_system_prompt(**kwargs: Any) -> str:
    """Generate system prompt for answer generation.

    Creates a structured prompt that instructs the LLM to generate
    informative answers based on context with JSON output.

    Args:
        **kwargs: Optional parameters for answer generation:
            topic (str): Optional conference topic for additional context.

    Returns:
        System prompt string for answer generation task.

    Example:
        >>> prompt = _generate_system_prompt(topic="Cloud Architecture")
    """
    topic = kwargs.get("topic")

    # Base system prompt
    base_prompt = """You are an expert technical advisor at a professional conference.

Your task is to provide comprehensive, accurate, and helpful answers to questions based on:
1. Relevant context from previous Q&A sessions (provided as "Context" in user message)
2. Your domain expertise and general knowledge
3. Best practices and industry standards

When answering:
- Prioritize information from the provided Context Q&A pairs when relevant
- Supplement with your domain knowledge when context is insufficient
- Clearly distinguish between information from context vs general knowledge"""

    # Add topic context if provided
    if topic:
        topic_context = (
            f"\n\nCurrent conference topic: {topic}\n"
            f"Ensure your answers are relevant to this topic when applicable."
        )
    else:
        topic_context = ""

    # Instructions for answer generation with clear sources_used rules
    instructions = """

Instructions:
1. Analyze the provided context Q&A pairs for relevant information
2. Synthesize information from multiple sources when applicable
3. Provide clear, structured, and actionable answers
4. Include specific examples or recommendations when relevant
5. Track which sources you used for the answer

Output format:
You MUST respond with ONLY a valid JSON object in this exact format:
{
    "answer": "Your comprehensive answer here. Use \\n for line breaks if needed.",
    "confidence": 0.95,
    "sources_used": ["context", "domain_knowledge"]
}

Field definitions:
- "answer": Your complete response to the question (required)
- "confidence": Float between 0.0 and 1.0 indicating answer quality (optional)
- "sources_used": Array indicating information sources (optional):
  * Include "context" if you used ANY information from the provided Q&A pairs
  * Include "domain_knowledge" if you used ANY of your general knowledge
  * Use ["context"] when answer is based ONLY on provided Q&A pairs
  * Use ["domain_knowledge"] when answer is based ONLY on your knowledge
  * Use ["context", "domain_knowledge"] when combining both sources

CRITICAL JSON RULES:
- NEVER use triple quotes in JSON
- NEVER use raw line breaks inside string values
- Use \\n for line breaks within strings
- All strings must be enclosed in single double quotes (")
- The entire response must be valid JSON that can be parsed by json.loads()
- Do not include any text, markdown, or explanations outside the JSON object

Example of CORRECT format:
{"answer": "Based on the context\\nHere is my answer", "confidence": 0.9, "sources_used": ["context"]}"""

    prompt = base_prompt + topic_context + instructions

    return prompt


def _generate_chat_history(**kwargs: Any) -> List[Dict[str, str]]:
    """Generate chat history (placeholder for future enhancement).

    Currently returns empty history as per requirements.
    Kept for potential future use and API consistency.

    Args:
        **kwargs: Reserved for future parameters.

    Returns:
        Empty list of message dictionaries.
    """
    # Placeholder function as requested
    # Can be enhanced in future for multi-turn conversations
    return []


def update_system_prompt(**kwargs: Any) -> str:
    """Update or retrieve the cached system prompt.

    Updates the global system prompt if kwargs are provided,
    otherwise returns the existing cached prompt.

    Args:
        **kwargs: Parameters for prompt generation:
            topic (str): Optional conference topic.

    Returns:
        The current system prompt string.

    Raises:
        ValueError: If system prompt is not initialized when needed.
    """
    global _system_prompt, _current_topic

    if kwargs:
        # Cache topic for reference
        if "topic" in kwargs:
            _current_topic = kwargs.get("topic")
            logger.debug(f"Conference topic set: {_current_topic}")

        _system_prompt = _generate_system_prompt(**kwargs)
        logger.debug("System prompt updated")
    elif _system_prompt is None:
        # Generate default prompt without topic
        _system_prompt = _generate_system_prompt()
        logger.debug("System prompt initialized with defaults")

    return _system_prompt


def update_chat_history(**kwargs: Any) -> List[Dict[str, str]]:
    """Update or retrieve the cached chat history.

    Placeholder implementation as per requirements.
    Returns empty history for single-turn Q&A.

    Args:
        **kwargs: Reserved for future parameters.

    Returns:
        Empty list (no chat history for current implementation).
    """
    global _chat_history

    # Always return empty as per requirements
    _chat_history = _generate_chat_history(**kwargs)
    return _chat_history.copy()


def get_messages(
    user_question: str, qa_pairs: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """Build complete message list for LLM request.

    Args:
        user_question: The question to answer.
        qa_pairs: Context Q&A pairs from previous conferences.

    Returns:
        List of message dictionaries formatted for LLM API.

    Raises:
        ValueError: If qa_pairs is invalid or empty.
    """
    messages_list = []

    # Get system prompt from cache
    system_prompt = update_system_prompt()
    if system_prompt:
        messages_list.append({"role": "system", "content": system_prompt})

    # Get chat history (empty for now)
    history = update_chat_history()
    messages_list.extend(history)

    # Format and add user prompt with Q&A context
    user_prompt = _format_user_prompt(user_question, qa_pairs)
    messages_list.append({"role": "user", "content": user_prompt})

    return messages_list


def _parse_json_response(response_text: str) -> Dict[str, Any]:
    """Parse and validate JSON response from LLM.

    Uses json.JSONDecoder.raw_decode() to safely extract JSON object
    from response text, handling cases where LLM might add extra text.

    Args:
        response_text: Raw text response from LLM.

    Returns:
        Parsed JSON as dictionary.

    Raises:
        ValueError: If response is not valid JSON or missing required fields.
    """
    try:
        # Clean the response text
        text = response_text.strip()

        # First attempt: try to fix common LLM mistakes
        # Replace triple quotes with regular quotes
        if '"""' in text:
            logger.warning("Found triple quotes in response, attempting to fix")
            # Extract content between triple quotes and escape it properly
            import re

            # Pattern to match JSON with triple-quoted strings
            pattern = r'(\{[^}]*"answer"\s*:\s*)"""(.*?)"""'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                # Replace newlines with \n and escape quotes
                content = match.group(2)
                content = content.replace("\n", "\\n").replace('"', '\\"')
                text = (
                    text[: match.start(2) - 3]
                    + '"'
                    + content
                    + '"'
                    + text[match.end(2) + 3 :]
                )
                logger.debug("Attempted to fix triple quotes")

        # Find the start of JSON object
        start_idx = text.find("{")
        if start_idx == -1:
            raise ValueError("No JSON object found in response")

        # Use JSONDecoder to properly parse JSON
        decoder = json.JSONDecoder()
        result, end_idx = decoder.raw_decode(text[start_idx:])

        logger.debug(
            f"Successfully extracted JSON from position {start_idx} to {start_idx + end_idx}"
        )

        # Validate required fields
        if not isinstance(result, dict):
            raise ValueError("Response must be a JSON object")

        if "answer" not in result:
            raise ValueError("Response missing 'answer' field")

        # Validate optional fields if present
        if "confidence" in result:
            confidence = result["confidence"]
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                logger.warning(f"Invalid confidence value: {confidence}, ignoring")
                result.pop("confidence", None)

        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from response: {response_text[:200]}...")
        # Try one more recovery attempt - extract just the answer
        try:
            import re

            # Try to extract answer content more aggressively
            answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
            if answer_match:
                logger.warning("Using fallback extraction for answer field")
                return {"answer": answer_match.group(1).replace("\\n", "\n")}
        except Exception:
            pass
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        logger.error(f"Failed to extract JSON from response: {response_text[:200]}...")
        raise ValueError(f"Failed to parse response: {e}")


def run(
    user_question: str,
    qa_pairs: List[Dict[str, str]],
    custom_params: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """Generate answer for a question using context Q&A pairs.

    Args:
        user_question: Question to answer.
        qa_pairs: List of Q&A dictionaries for context. Each dictionary must have:
            - question (str): Previous question
            - answer (str): Previous answer
        custom_params: Optional parameter overrides for this request.

    Returns:
        Tuple containing:
            - Valid JSON string with answer result containing:
                - answer: The generated answer
                - confidence: Optional confidence score (0.0 to 1.0)
                - sources_used: Optional list of sources used
            - List of messages sent to the LLM

    Raises:
        ValueError: If qa_pairs is empty/invalid or response format is invalid.
        RuntimeError: If LLM response format is unexpected or streaming is requested.

    Example:
        >>> qa_pairs = [
        ...     {"question": "How to scale?", "answer": "Use load balancers..."}
        ... ]
        >>> result, messages = run("What about caching?", qa_pairs)
        >>> print(result)
        '{"answer": "Based on scaling context, caching can help...", "confidence": 0.9}'
    """
    # Validate qa_pairs
    if not qa_pairs:
        raise ValueError("qa_pairs cannot be empty - context is required for RAG")

    for i, pair in enumerate(qa_pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"qa_pairs[{i}] must be a dictionary")
        if "question" not in pair or "answer" not in pair:
            raise ValueError(f"qa_pairs[{i}] must contain 'question' and 'answer' keys")

    # Merge custom parameters with defaults
    request_params = {k: v for k, v in params.items() if v is not None}
    if custom_params:
        for k, v in custom_params.items():
            if v is not None:
                request_params[k] = v

    # Answer generation requires non-streaming mode
    if request_params.get("stream", False):
        raise RuntimeError(
            "Streaming not supported for answer generation. "
            "Structured JSON output requires non-streaming mode."
        )

    # Build messages with context
    messages_list = get_messages(user_question, qa_pairs)

    logger.debug(f"Generating answer for: {user_question[:100]}...")
    logger.debug(f"Using {len(qa_pairs)} Q&A pairs for context")
    logger.debug(f"Request params: {request_params}")

    try:
        # Make LLM request
        response = llm.chat_completion(messages=messages_list, **request_params)

        # Safely extract content from response
        if not isinstance(response, dict):
            raise RuntimeError(f"Expected dict response, got {type(response)}")

        if "choices" not in response:
            raise RuntimeError("Response missing 'choices' field")

        choices = response["choices"]
        if not isinstance(choices, list) or len(choices) == 0:
            raise RuntimeError("Response 'choices' is empty or invalid")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError(f"Expected dict in choices[0], got {type(first_choice)}")

        if "message" not in first_choice:
            raise RuntimeError("Response choice missing 'message' field")

        message = first_choice["message"]
        if not isinstance(message, dict) or "content" not in message:
            raise RuntimeError("Response message missing 'content' field")

        response_text = message["content"]

        # Parse and validate JSON response
        parsed_result = _parse_json_response(response_text)

        # Return as formatted JSON string with messages
        result_json = json.dumps(parsed_result, ensure_ascii=False, indent=2)
        return result_json, messages_list

    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        raise


# Test section
if __name__ == "__main__":
    """Test the answer generation module for RAG system."""

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s][%(filename)s:%(lineno)d][%(message)s]",
    )

    print("=== Answer Generation Module Test (RAG System) ===\n")

    # Test 1: Initialize without topic
    try:
        print("Test 1: Initialize answer generation system without topic")

        # Initialize the system without topic
        prompt = update_system_prompt()
        print(f"✔ System initialized without topic")
        print(f"✔ Prompt length: {len(prompt)} characters\n")

    except Exception as e:
        logger.error(f"Test 1 failed: {e}")
        raise

    # Test 2: Initialize with conference topic
    try:
        print("Test 2: Initialize with conference topic")

        test_topic = "Cloud Native Architecture and Microservices"

        # Initialize with topic
        prompt = update_system_prompt(topic=test_topic)
        print(f"✔ System initialized with topic: {test_topic}")
        print(f"✔ Prompt updated successfully")

        # Verify topic is in prompt
        if test_topic in prompt:
            print(f"✔ Topic correctly included in system prompt\n")

    except Exception as e:
        logger.error(f"Test 2 failed: {e}")
        raise

    # Test 3: Generate answer with minimal context
    try:
        print("Test 3: Generate answer with minimal context")

        # Minimal Q&A context
        minimal_qa = [
            {
                "question": "What are the benefits of using Docker?",
                "answer": "Docker provides consistency across environments, faster deployment, and resource efficiency.",
            }
        ]

        question = "How does containerization help with scaling?"

        print("\n--- INPUT DATA ---")
        print(f"Question: {question}")
        print(f"Context Q&A pairs count: {len(minimal_qa)}")
        print("Q&A pairs content:")
        for i, qa in enumerate(minimal_qa, 1):
            print(f"  Pair {i}:")
            print(f"    Q: {qa['question']}")
            print(f"    A: {qa['answer'][:80]}...")

        try:
            result_json, messages = run(question, minimal_qa)
            result = json.loads(result_json)

            print("\n--- OUTPUT DATA ---")
            print("Returned JSON structure:")
            print(f"  Fields: {list(result.keys())}")
            print("\nParsed result content:")
            print(f"  answer: {result['answer'][:150]}...")
            if "confidence" in result:
                print(f"  confidence: {result['confidence']}")
            if "sources_used" in result:
                print(f"  sources_used: {result['sources_used']}")

            print("\n--- MESSAGES SENT TO LLM ---")
            print(f"Total messages: {len(messages)}")
            for i, msg in enumerate(messages, 1):
                print(f"Message {i}:")
                print(f"  Role: {msg['role']}")
                print(f"  Content length: {len(msg['content'])} chars")
                if msg["role"] == "system":
                    print(f"  Content preview: {msg['content'][:100]}...")
                elif msg["role"] == "user":
                    # Show structure of user message
                    if "Context" in msg["content"]:
                        print(f"  Contains context Q&A pairs: Yes")
                    if question in msg["content"]:
                        print(f"  Contains current question: Yes")

            print("\n✔ Test completed successfully")

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            print(f"✗ Answer generation failed: {e}")

        print("\n" + "=" * 50 + "\n")

    except Exception as e:
        logger.error(f"Test 3 failed: {e}")
        raise

    # Test 4: Generate answer with rich context
    try:
        print("Test 4: Generate answer with rich context")

        # Rich Q&A context
        rich_qa = [
            {
                "question": "How do you implement CI/CD for microservices?",
                "answer": "We use GitLab CI with separate pipelines per service, automated testing, and Kubernetes deployments.",
            },
            {
                "question": "What's your monitoring strategy?",
                "answer": "We implement the three pillars: metrics with Prometheus, logs with ELK, and traces with Jaeger.",
            },
            {
                "question": "How do you handle service discovery?",
                "answer": "We use Consul for service registry and health checking, integrated with our load balancers.",
            },
        ]

        question = "What are the key considerations for microservices in production?"

        print("\n--- INPUT DATA ---")
        print(f"Question: {question}")
        print(f"Context Q&A pairs count: {len(rich_qa)}")
        print("Q&A pairs content:")
        for i, qa in enumerate(rich_qa, 1):
            print(f"  Pair {i}:")
            print(f"    Q: {qa['question']}")
            print(f"    A: {qa['answer']}")

        try:
            result_json, messages = run(question, rich_qa)
            result = json.loads(result_json)

            print("\n--- OUTPUT DATA ---")
            print("Raw JSON response:")
            print(result_json[:300] + "..." if len(result_json) > 300 else result_json)

            print("\nParsed result:")
            print(f"  Type: {type(result)}")
            print(f"  Fields present: {list(result.keys())}")
            print(f"  Answer length: {len(result['answer'])} characters")

            if "confidence" in result:
                print(f"  Confidence value: {result['confidence']}")
            if "sources_used" in result:
                print(f"  Sources used: {result['sources_used']}")

            print("\n--- MESSAGES ANALYSIS ---")
            print(f"Total messages sent to LLM: {len(messages)}")

            # Analyze user message
            user_message = next((m for m in messages if m["role"] == "user"), None)
            if user_message:
                user_content = user_message["content"]
                context_count = user_content.count("Context")
                print(f"Context Q&A pairs in prompt: {context_count}")
                print(f"User message total length: {len(user_content)} chars")

                # Check what's included
                print("User message contains:")
                print(
                    f"  - Current question: {'Yes' if question in user_content else 'No'}"
                )
                print(
                    f"  - Context header: {'Yes' if 'relevant Q&A pairs' in user_content else 'No'}"
                )
                print(
                    f"  - All Q&A pairs: {'Yes' if all(qa['question'] in user_content for qa in rich_qa) else 'No'}"
                )

            # Analyze system message
            system_message = next((m for m in messages if m["role"] == "system"), None)
            if system_message:
                sys_content = system_message["content"]
                print(f"\nSystem message length: {len(sys_content)} chars")
                print(f"System message contains:")
                print(
                    f"  - JSON instructions: {'Yes' if 'JSON' in sys_content else 'No'}"
                )
                print(
                    f"  - Conference topic: {'Yes' if _current_topic and _current_topic in sys_content else 'No'}"
                )

            print("\n✔ Test completed successfully")

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            print(f"\n✗ Answer generation failed: {e}")
            print(f"Error type: {type(e).__name__}")

        print("\n" + "=" * 50 + "\n")

    except Exception as e:
        logger.error(f"Test 4 failed: {e}")
        raise

    # Test 5: Error handling
    try:
        print("Test 5: Error handling")

        print("\n1. Test with empty qa_pairs:")
        try:
            result, messages = run("Test question", [])
            print("✗ Should have raised ValueError")
        except ValueError as e:
            print(f"✔ Correctly raised ValueError: {str(e)[:50]}...")

        print("\n2. Test with invalid qa_pairs format:")
        try:
            invalid_qa = [{"question": "Only question, no answer"}]
            result, messages = run("Test question", invalid_qa)
            print("✗ Should have raised ValueError")
        except ValueError as e:
            print(f"✔ Correctly raised ValueError for missing 'answer' key")

        print("\n3. Test with streaming (should fail):")
        try:
            valid_qa = [{"question": "Q", "answer": "A"}]
            result, messages = run("Test", valid_qa, custom_params={"stream": True})
            print("✗ Should have raised RuntimeError")
        except RuntimeError as e:
            print(f"✔ Correctly raised RuntimeError: {str(e)[:50]}...")

        print("\n" + "=" * 50 + "\n")

    except Exception as e:
        logger.error(f"Test 5 failed: {e}")
        raise

    # Test 6: Verify message structure
    try:
        print("Test 6: Verify message structure and data flow")

        test_qa = [
            {
                "question": "What is Kubernetes?",
                "answer": "Container orchestration platform.",
            }
        ]

        question = "How to deploy to Kubernetes?"

        print("\n--- INPUT PARAMETERS ---")
        print(f"run() function inputs:")
        print(f"  user_question: '{question}'")
        print(f"  qa_pairs: {test_qa}")
        print(f"  custom_params: None")

        result_json, messages = run(question, test_qa)

        print("\n--- OUTPUT PARAMETERS ---")
        print(f"run() function outputs:")
        print(f"  result_json type: {type(result_json)}")
        print(f"  result_json length: {len(result_json)} chars")
        print(f"  messages type: {type(messages)}")
        print(f"  messages length: {len(messages)} items")

        # Parse and display result
        result = json.loads(result_json)
        print("\nParsed result_json content:")
        for key, value in result.items():
            if key == "answer":
                print(f"  {key}: '{value[:100]}...' (truncated)")
            else:
                print(f"  {key}: {value}")

        print("\n--- DETAILED MESSAGE STRUCTURE ---")
        for i, msg in enumerate(messages):
            print(f"\nMessage {i + 1}:")
            print(f"  Type: {type(msg)}")
            print(f"  Keys: {list(msg.keys())}")
            print(f"  Role: '{msg['role']}'")
            print(f"  Content type: {type(msg['content'])}")
            print(f"  Content length: {len(msg['content'])} chars")

            # Detailed content analysis
            content = msg["content"]
            if msg["role"] == "system":
                print("  System message analysis:")
                print(f"    - Contains 'JSON': {'Yes' if 'JSON' in content else 'No'}")
                print(
                    f"    - Contains 'Output format': {'Yes' if 'Output format' in content else 'No'}"
                )
                print(
                    f"    - Contains topic: {'Yes' if _current_topic and _current_topic in content else 'No'}"
                )
                print(f"    - First 150 chars: {content[:150]}...")
            elif msg["role"] == "user":
                print("  User message analysis:")
                print(
                    f"    - Contains question: {'Yes' if question in content else 'No'}"
                )
                print(
                    f"    - Contains 'Context': {'Yes' if 'Context' in content else 'No'}"
                )
                print(
                    f"    - Contains Q&A pairs: {'Yes' if test_qa[0]['question'] in content else 'No'}"
                )

                # Extract structure
                lines = content.split("\n")
                print(f"    - Total lines: {len(lines)}")
                print(f"    - Structure preview:")
                for j, line in enumerate(lines[:10]):  # First 10 lines
                    if line.strip():
                        print(f"        Line {j+1}: {line[:60]}...")

        print("\n--- VALIDATION ---")
        # Validate the structure
        validations = [
            ("Has system message", any(m["role"] == "system" for m in messages)),
            ("Has user message", any(m["role"] == "user" for m in messages)),
            ("Result has 'answer' field", "answer" in result),
            ("Answer is non-empty", bool(result.get("answer", "").strip())),
            ("Messages list is proper type", isinstance(messages, list)),
            (
                "All messages have required keys",
                all("role" in m and "content" in m for m in messages),
            ),
        ]

        for check_name, check_result in validations:
            status = "✔" if check_result else "✗"
            print(f"{status} {check_name}: {check_result}")

        print("\n✔ Test 6 completed successfully")
        print("\n" + "=" * 50 + "\n")

    except Exception as e:
        logger.error(f"Test 6 failed: {e}")
        print(f"\n✗ Test 6 failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        raise

    print("\n=== All tests completed successfully ===")

    # Bonus: Complete data flow demonstration
    print("\n" + "=" * 50)
    print("BONUS: Complete Data Flow Demonstration")
    print("=" * 50 + "\n")

    try:
        print("This demonstrates the complete input → processing → output flow\n")

        # Setup
        demo_topic = "Python Best Practices"
        demo_qa = [
            {
                "question": "How to handle errors in Python?",
                "answer": "Use try-except blocks with specific exception types.",
            },
            {
                "question": "What's the best way to manage dependencies?",
                "answer": "Use virtual environments and requirements.txt or poetry.",
            },
        ]
        demo_question = "How should I structure a Python project?"

        print("STEP 1: Initialize with topic")
        print(f"  Topic: '{demo_topic}'")
        update_system_prompt(topic=demo_topic)

        print("\nSTEP 2: Prepare inputs")
        print(f"  Question: '{demo_question}'")
        print(f"  Context Q&A pairs: {len(demo_qa)} pairs")
        for i, qa in enumerate(demo_qa, 1):
            print(f"    [{i}] Q: {qa['question'][:50]}...")
            print(f"        A: {qa['answer'][:50]}...")

        print("\nSTEP 3: Call run() function")
        print(f"  run(user_question={demo_question!r},")
        print(f"      qa_pairs=[...{len(demo_qa)} pairs...],")
        print(f"      custom_params=None)")

        # Execute
        result_json, messages = run(demo_question, demo_qa)
        result = json.loads(result_json)

        print("\nSTEP 4: Process outputs")
        print(f"  Returned tuple: (result_json, messages)")
        print(f"  - result_json: string of {len(result_json)} chars")
        print(f"  - messages: list of {len(messages)} message dicts")

        print("\nSTEP 5: Parse and use results")
        print(f"  Parsed JSON fields: {list(result.keys())}")
        print(f"  Answer (first 200 chars): {result['answer'][:200]}...")
        if "confidence" in result:
            print(f"  Confidence score: {result['confidence']}")

        print("\n✔ Data flow demonstration complete!")

    except Exception as e:
        print(f"✗ Demo failed: {e}")

    print("\n" + "=" * 50)
