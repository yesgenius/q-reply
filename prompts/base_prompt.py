"""Base prompt template for LLM interactions with caching.

This module provides a universal template for creating prompt modules
with consistent structure, error handling, caching, and testing capabilities.

The module implements caching for system prompts and chat history to avoid
redundant computations while maintaining flexibility for dynamic updates.

Example:
    To create a new prompt, copy this file and modify:
    1. Update module docstring
    2. Change llm initialization if needed (different provider)
    3. Modify params dictionary for model-specific settings
    4. Set initial _system_prompt and _chat_history if needed
    5. Implement _generate_system_prompt() for dynamic updates
    6. Implement _generate_chat_history() for dynamic updates
    7. Update test section with relevant examples

Usage:
    ```python
    from prompts import base_prompt

    # Use with pre-initialized prompt
    response = base_prompt.run("Explain recursion")
    print(base_prompt.messages)  # View formatted JSON of messages

    # Or update dynamically
    base_prompt.update_system_prompt(role="teacher")
    response = base_prompt.run("Explain recursion")
    ```
    python -m prompts.base_prompt

"""

from __future__ import annotations

from collections.abc import (
    Generator,
    Generator as GeneratorType,
)
import json
import logging
from typing import Any, cast

from gigachat.client import GigaChatClient
from utils.logger import get_logger


logger = get_logger(__name__)

# Initialize LLM client (can be replaced with other providers)
llm = GigaChatClient()

# Default model parameters
params: dict[str, Any] = {
    "model": "GigaChat",
    "temperature": None,
    "top_p": None,
    "stream": False,
    "max_tokens": None,
    "repetition_penalty": None,
}

# Global cached variables with optional initialization
# For base_prompt.py: keep as None/empty for lazy initialization
# For specific prompts: can initialize with static values
# _system_prompt: Optional[str] = "You are a specialized assistant..."
_system_prompt: str | None = None  # Or: "You are a specialized assistant..."
_chat_history: list[dict[str, str]] = []  # Or: [{"role": "system", "content": "..."}]

# Public variable to store formatted messages as JSON string
messages: str = "[]"

# Example of static initialization (uncomment in specific prompt modules):
# _system_prompt = """You are an expert Python developer and teacher.
# Your responses should be clear, educational, and include code examples.
# Always explain complex concepts in simple terms."""
#
# _chat_history = [
#     {"role": "user", "content": "I want to learn Python"},
#     {"role": "assistant", "content": "Great! I'll help you learn Python step by step."}
# ]


def _update_messages(messages_list: list[dict[str, str]]) -> None:
    """Update the public messages variable with formatted JSON.

    This is an internal function that updates the module-level messages
    variable with a pretty-printed JSON representation of the messages list.

    Args:
        messages_list: List of message dictionaries with 'role' and 'content' keys.

    Note:
        This function is called automatically in get_messages() to ensure
        the public messages variable always reflects the current state.
    """
    global messages
    messages = json.dumps(messages_list, indent=2, ensure_ascii=False)
    logger.debug(f"Updated messages variable with {len(messages_list)} messages")


def _generate_system_prompt(**kwargs: Any) -> str:
    """Generate system prompt with dynamic context.

    This is an internal method that creates the system prompt.
    Override this in specific prompt implementations to provide
    custom system prompts with dynamic data injection.

    Note:
        This function is called only when:
        1. No cached prompt exists and update_system_prompt() is called
        2. update_system_prompt() is called with kwargs

    Args:
        **kwargs: Dynamic data for prompt generation.
            Common keys might include:
            - context: Additional context information
            - role: Specific role for the assistant
            - constraints: Any constraints or rules

    Returns:
        System prompt string.

    Example:
        >>> prompt = _generate_system_prompt(role="teacher", subject="math")
        >>> print(prompt)
        'You are a professional teacher specializing in math.'
    """
    # Default implementation - override in specific prompts
    base_prompt = "You are a helpful AI assistant."

    # Example of dynamic data injection
    if "role" in kwargs:
        base_prompt = f"You are a professional {kwargs['role']}."

    if "context" in kwargs:
        base_prompt += f" Context: {kwargs['context']}"

    return base_prompt


def _generate_chat_history(**kwargs: Any) -> list[dict[str, str]]:
    """Generate chat history with dynamic content.

    This is an internal method that creates the chat history.
    Override this to provide specific chat history patterns
    or to load history from external sources.

    Note:
        This function is called only when update_chat_history()
        is called with kwargs. If _chat_history is pre-initialized,
        this may never be called.

    Args:
        **kwargs: Dynamic data for history generation.
            Common keys might include:
            - session_id: ID to load specific session history
            - last_n_messages: Number of previous messages to include
            - include_examples: Whether to include example exchanges
            - append: Whether to append to existing history

    Returns:
        List of message dictionaries with 'role' and 'content' keys.

    Example:
        >>> history = _generate_chat_history(include_examples=True)
        >>> print(len(history))
        2
    """
    # Default implementation - override in specific prompts
    history: list[dict[str, str]] = []

    # Option to append to existing history
    if kwargs.get("append", False):
        history = _chat_history.copy()

    # Example: Add example exchanges if requested
    if kwargs.get("include_examples", False):
        history.extend(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello! How can I assist you today?"},
            ]
        )

    # TODO: Implement actual history loading from database/session
    # Example placeholder for session-based history:
    # if session_id := kwargs.get("session_id"):
    #     history.extend(load_history_from_db(session_id))

    return history


def update_system_prompt(**kwargs: Any) -> str:
    """Update or retrieve the cached system prompt.

    Updates the global system prompt if kwargs are provided,
    otherwise returns the existing cached prompt. If no cached
    prompt exists and no kwargs provided, generates a default one.

    Args:
        **kwargs: Dynamic data for prompt generation.
            If empty, returns existing cached prompt.

    Returns:
        The current system prompt string.

    Example:
        >>> # Generate and cache new prompt
        >>> prompt = update_system_prompt(role="expert")
        >>> print(prompt)
        'You are a professional expert.'

        >>> # Retrieve cached prompt (or pre-initialized)
        >>> cached = update_system_prompt()
        >>> print(cached)
        'You are a professional expert.'
    """
    global _system_prompt

    if kwargs:
        # Generate new prompt if kwargs provided
        _system_prompt = _generate_system_prompt(**kwargs)
        logger.debug(f"System prompt updated with kwargs: {list(kwargs.keys())}")
    elif _system_prompt is None:
        # Generate default only if not pre-initialized
        _system_prompt = _generate_system_prompt()
        logger.debug("Default system prompt generated")
    else:
        # Use pre-initialized or previously cached prompt
        logger.debug("Using cached/pre-initialized system prompt")

    # Ensure we always return a string for type consistency
    assert _system_prompt is not None, "System prompt should never be None at this point"
    return _system_prompt


def update_chat_history(**kwargs: Any) -> list[dict[str, str]]:
    """Update or retrieve the cached chat history.

    Updates the global chat history if kwargs are provided,
    otherwise returns the existing cached history. If no cached
    history exists and no kwargs provided, returns empty list
    or pre-initialized history.

    Args:
        **kwargs: Dynamic data for history generation.
            Special keys:
            - append: If True, appends to existing history
            - clear: If True, clears history before updating

    Returns:
        The current chat history list (copy).

    Example:
        >>> # Generate and cache new history
        >>> history = update_chat_history(include_examples=True)
        >>> print(len(history))
        2

        >>> # Append to existing history
        >>> history = update_chat_history(append=True, include_examples=True)
        >>> print(len(history))
        4
    """
    global _chat_history

    if kwargs:
        local_kwargs = dict(kwargs)
        if local_kwargs.pop("clear", False):
            _chat_history = []
            logger.debug("Chat history cleared")

        _chat_history = _generate_chat_history(**local_kwargs)
        logger.debug(f"Chat history updated with kwargs: {list(local_kwargs.keys())}")
    else:
        logger.debug(f"Using cached/pre-initialized chat history (length: {len(_chat_history)})")

    return _chat_history.copy()  # Return copy to prevent external modifications


def add_to_chat_history(message: dict[str, str]) -> None:
    """Add a single message to chat history.

    Utility function to append messages to existing history
    without regenerating the entire history.

    Args:
        message: Dictionary with 'role' and 'content' keys.

    Example:
        >>> add_to_chat_history({"role": "user", "content": "Hello"})
        >>> add_to_chat_history({"role": "assistant", "content": "Hi!"})
    """
    global _chat_history
    _chat_history.append(message)
    logger.debug(f"Added {message['role']} message to chat history")


def clear_cache() -> None:
    """Clear all cached data.

    Resets both system prompt and chat history caches.
    Also clears the public messages variable.
    Useful for starting fresh sessions or switching contexts.

    Note:
        If prompts were pre-initialized at module level,
        this will clear them. Use with caution in modules
        with static initialization.

    Example:
        >>> update_system_prompt(role="teacher")
        >>> clear_cache()
        >>> prompt = update_system_prompt()  # Will generate default
        >>> print("teacher" in prompt)
        False
    """
    global _system_prompt, _chat_history, messages
    _system_prompt = None
    _chat_history = []
    messages = "[]"
    logger.debug("Cache cleared")


def reset_to_defaults() -> None:
    """Reset to module-level default values.

    Unlike clear_cache(), this resets to the original
    module-level initialized values, useful when the module
    has pre-defined static prompts. Also resets the messages variable.

    Example:
        >>> # In a module with pre-initialized prompt
        >>> update_system_prompt(role="teacher")
        >>> reset_to_defaults()
        >>> # Returns to original module-level prompt
    """
    global _system_prompt, _chat_history, messages

    # Reset to module defaults
    # This works because Python evaluates the right side first
    _system_prompt = globals().get("_DEFAULT_SYSTEM_PROMPT", None).copy()
    _chat_history = globals().get("_DEFAULT_CHAT_HISTORY", []).copy()
    messages = "[]"

    logger.debug("Reset to module defaults")


def get_messages(user_question: str) -> list[dict[str, str]]:
    """Build complete message list for LLM request.

    Combines cached system prompt, chat history, and user question
    into a properly formatted message list. Uses cached values
    when available. Automatically updates the public messages variable.

    Args:
        user_question: The current user question/input.

    Returns:
        List of message dictionaries formatted for LLM API.

    Example:
        >>> update_system_prompt(role="assistant")
        >>> messages_list = get_messages("What is Python?")
        >>> print(messages_list[0]["role"])
        'system'
        >>> print(messages_list[-1]["content"])
        'What is Python?'
        >>> # The global 'messages' variable now contains JSON string
        >>> import json
        >>> parsed = json.loads(messages)
        >>> print(parsed[0]["role"])
        'system'
    """
    messages_list = []

    # Get system prompt from cache or generate default
    system_prompt = update_system_prompt()
    if system_prompt:
        messages_list.append({"role": "system", "content": system_prompt})

    # Get chat history from cache
    history = update_chat_history()
    messages_list.extend(history)

    # Add current user question
    messages_list.append({"role": "user", "content": user_question})

    # Update the public messages variable with formatted JSON
    _update_messages(messages_list)

    return messages_list


def run(
    user_question: str, custom_params: dict[str, Any] | None = None
) -> str | Generator[dict[str, Any], None, None]:
    """Execute prompt with the configured LLM.

    Uses cached system prompt and chat history to build messages,
    then executes the LLM request with the user's question.
    Automatically updates the public messages variable with the
    exact messages sent to the LLM.

    Args:
        user_question: User input to generate response for.
        custom_params: Optional parameter overrides for this request.

    Returns:
        Assistant's response as string or generator for streaming.

    Raises:
        RuntimeError: If LLM response format is unexpected.
        Exception: Any exception from the LLM client.

    Example:
        >>> # With pre-initialized or cached context
        >>> response = run("Explain recursion")
        >>> print(type(response))
        <class 'str'>
        >>> print(messages)  # Shows formatted JSON of messages sent

        >>> # With custom parameters
        >>> response = run("Be brief", {"max_tokens": 50})
    """
    # Merge custom parameters with defaults
    request_params = {k: v for k, v in params.items() if v is not None}
    if custom_params:
        for k, v in custom_params.items():
            if v is not None:
                request_params[k] = v

    # Build messages using cached data
    # This also updates the public messages variable
    messages_list = get_messages(user_question)

    # Log request details for debugging
    logger.debug(f"Request params: {request_params}")
    logger.debug(f"Message count: {len(messages_list)}")

    try:
        # Make LLM request
        response = llm.chat_completion(messages=messages_list, **request_params)

        # Handle response based on stream setting
        if request_params.get("stream", False):
            # For streaming, return generator as-is
            if isinstance(response, GeneratorType):
                return response
            raise RuntimeError("Expected generator for streaming response")
        # For non-streaming, extract content from response
        if isinstance(response, dict) and "choices" in response:
            return cast("str", response["choices"][0]["message"]["content"])
        raise RuntimeError(f"Unexpected response format: {type(response)}")

    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        raise


# Store defaults for reset functionality (optional)
_DEFAULT_SYSTEM_PROMPT = _system_prompt
_DEFAULT_CHAT_HISTORY = _chat_history.copy() if _chat_history else []


# Test section
if __name__ == "__main__":
    """Test the prompt module with various scenarios."""

    logging.basicConfig(
        level=logging.INFO,  # DEBUG
        format="[%(asctime)s][%(name)s][%(levelname)s][%(filename)s:%(lineno)d][%(message)s]",
    )

    print("=== Improved Base Prompt Module Test ===\n")

    # Test 1: Basic usage with potential pre-initialized values
    try:
        print("Test 1: Basic prompt execution with messages tracking")
        # Check if we have pre-initialized prompt
        current_prompt = update_system_prompt()
        print(f"Current system prompt: {current_prompt[:50]}...")

        question = "What is artificial intelligence?"
        response = run(question)
        print(f"Q: {question}")
        print(f"A: {response[:200] if isinstance(response, str) else 'Response'}...")
        print(f"Messages sent to LLM:\n{messages[:300]}...\n")
    except Exception as e:
        logger.error(f"Test 1 failed: {e}")

    # Test 2: Dynamic update of system prompt
    try:
        print("Test 2: Dynamic system prompt update")
        # Update the system prompt
        update_system_prompt(role="programming teacher", context="Teaching Python to beginners")

        question = "How do functions work?"
        response = run(question)
        print(f"Q: {question}")
        print(f"A: {response[:200] if isinstance(response, str) else 'Response'}...")
        print(f"Updated messages:\n{messages[:300]}...\n")
    except Exception as e:
        logger.error(f"Test 2 failed: {e}")

    # Test 3: Working with chat history
    try:
        print("Test 3: Chat history management")
        # Clear and set new history
        update_chat_history(clear=True, include_examples=True)

        # Add individual messages
        add_to_chat_history({"role": "user", "content": "I need help with Python"})
        add_to_chat_history(
            {"role": "assistant", "content": "I'd be happy to help you with Python!"}
        )

        history = update_chat_history()
        print(f"History length: {len(history)}")

        question = "Can you explain variables?"
        response = run(question)
        print(f"Q: {question}")
        print(f"A: {response[:200] if isinstance(response, str) else 'Response'}...")
        print(f"Messages with history:\n{messages[:500]}...\n")
    except Exception as e:
        logger.error(f"Test 3 failed: {e}")

    # Test 4: Cache clearing and reset
    try:
        print("Test 4: Cache management and messages clearing")
        print(f"Messages JSON string length before clear: {len(messages)} characters")
        print(f"_system_prompt:[{_system_prompt}]")

        # Clear all cached data
        clear_cache()
        print("Cache cleared")
        print(f"Messages after clear: {messages}")
        print(f"_system_prompt:`{_system_prompt}`")

        # This will generate default prompt
        prompt_after_clear = update_system_prompt()
        print(f"Prompt after clear: {prompt_after_clear[:50]}...")

        # Reset to defaults (if they were set)
        reset_to_defaults()
        print("Reset to defaults")
        print(f"Messages after reset: {messages}")
        print(f"_system_prompt:[{_system_prompt}]")

        prompt_after_reset = update_system_prompt()
        print(
            f"Prompt after reset: {prompt_after_reset[:50] if prompt_after_reset else 'None'}...\n"
        )
    except Exception as e:
        logger.error(f"Test 4 failed: {e}")

    # Test 5: Custom parameters
    try:
        print("Test 5: Custom parameters")
        question = "List three primary colors."
        response = run(question, custom_params={"temperature": 0.1, "max_tokens": 100})
        print(f"Q: {question}")
        print(f"A: {response}")
        print(f"Final messages state:\n{messages}\n")
    except Exception as e:
        logger.error(f"Test 5 failed: {e}")

    print("=== Tests completed ===")
