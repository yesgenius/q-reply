"""GigaChat API client implementation.

This module provides a client for interacting with the GigaChat API,
supporting both mTLS (certificate-based) and Basic authentication modes.

Usage:
    python -m gigachat.client

Models list:
- Embeddings: embedder
- Embeddings-2: embedder
- EmbeddingsGigaR: embedder
- GigaChat: chat
- GigaChat-2: chat
- GigaChat-2-Max: chat
- GigaChat-2-Pro: chat
- GigaChat-Max: chat
- GigaChat-Max-preview: chat
- GigaChat-Plus: chat
- GigaChat-Plus-preview: chat
- GigaChat-preview: chat
- GigaChat-Pro: chat
- GigaChat-Pro-preview: chat

Models list in contur:
- Embeddings: embedder
- Embeddings-2: embedder
- EmbeddingsGigaR: embedder
- GigaChat: chat
- GigaChat-2: chat
- GigaChat-2-Max: chat
- GigaChat-2-Pro: chat
- GigaChat-Max: chat
- GigaChat-Pro: chat
- SaluteEmbeddings: embedder
- SaluteEmbeddings_AB: embedder
- SaluteV5Embeddings: embedder
- SaluteV5Embeddings_AB: embedder

"""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
import json
import logging
import time
from typing import Any
import uuid

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from utils.logger import get_logger

from .config import GigaChatConfig


logger = get_logger(__name__)


class GigaChatClient:
    """Client for GigaChat API.

    This client supports two authentication modes:
    1. mTLS mode: When client certificates are provided, uses mutual TLS
       authentication without OAuth token exchange.
    2. Basic mode: When no certificates are provided, uses Basic auth to
       obtain OAuth token, then uses Bearer token for API calls.

    Attributes:
        config: Configuration object for the client.
        session: Requests session for connection pooling.
        access_token: Current access token (only used in Basic mode).
        token_expires_at: Token expiration timestamp (only used in Basic mode).
    """

    def __init__(self, config: GigaChatConfig | None = None):
        """Initialize GigaChat client.

        Args:
            config: Configuration object. If None, will load from environment.
        """
        logger.debug(f"Initializing GigaChatClient with config: {config}")
        self.config = config or GigaChatConfig()
        self.session = self._create_session()
        self.access_token: str | None = None
        self.token_expires_at: int | None = None

        # Log authentication mode
        auth_mode = "mTLS" if self.config.is_cert_auth() else "Basic"
        logger.info(f"GigaChatClient initialized in {auth_mode} authentication mode")

    def _create_session(self) -> requests.Session:
        """Create configured requests session with retry logic.

        Returns:
            Configured requests session.
        """
        logger.debug("Creating HTTP session with retry strategy")
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(
                ["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
                # ["HEAD", "GET", "OPTIONS"]  # to avoid repeated billing/duplication
            ),
            backoff_factor=1,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Configure SSL verification and certificates
        if self.config.is_cert_auth():
            # mTLS mode: use client certificates, disable server verification
            cert_paths = self.config.get_cert_paths()
            if cert_paths:
                session.cert = cert_paths
                logger.debug(f"Using client certificates: {cert_paths}")

            # Disable SSL verification for mTLS mode
            session.verify = False
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.debug("SSL verification disabled for mTLS authentication")
        else:
            # Basic mode: no client certificates, disable server verification
            session.verify = False
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.debug("SSL verification disabled for Basic authentication")

        logger.debug("HTTP session created successfully")
        return session

    def _is_token_valid(self) -> bool:
        """Check if current token is still valid.

        Note: Only relevant for Basic authentication mode.

        Returns:
            True if token is valid, False otherwise.
        """
        # In mTLS mode, we don't use tokens
        if self.config.is_cert_auth():
            return True

        if not self.access_token or not self.token_expires_at:
            logger.debug("Token invalid: no token or expiration time")
            return False

        # Add 60 second buffer before expiration
        current_time = int(time.time())
        is_valid = current_time < (self.token_expires_at - 60)

        if logger.isEnabledFor(logging.DEBUG):
            time_remaining = self.token_expires_at - current_time
            logger.debug(
                f"Token validity check: valid={is_valid}, time_remaining={time_remaining}s"
            )

        return is_valid

    def _safe_parse_expiration(self, token_data: dict[str, Any]) -> int | None:
        """Safely parse token expiration from response.

        Args:
            token_data: Token response data.

        Returns:
            Expiration timestamp in seconds, or None if parsing fails.
        """
        try:
            # Try to get expires_at first
            expires_at = token_data.get("expires_at")
            if expires_at is not None:
                # Ensure it's a number
                expires_at_num = int(expires_at)
                # If value > 10^12, it's likely in milliseconds
                if expires_at_num > 10**12:
                    return expires_at_num // 1000
                return expires_at_num
        except (TypeError, ValueError) as e:
            logger.debug(f"Failed to parse expires_at: {e}")

        try:
            # Fallback to expires_in (relative time in seconds)
            expires_in = token_data.get("expires_in")
            if expires_in is not None:
                expires_in_num = int(expires_in)
                return int(time.time()) + expires_in_num
        except (TypeError, ValueError) as e:
            logger.debug(f"Failed to parse expires_in: {e}")

        # Default to 30 minutes if no expiration info
        logger.warning("No valid expiration info in token response, defaulting to 30 minutes")
        return int(time.time()) + 1800

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers based on auth mode.

        Returns:
            Dictionary with authentication headers.

        Raises:
            ValueError: If authentication is not properly configured.
        """
        if self.config.is_cert_auth():
            # mTLS mode: no Authorization header needed
            return {}
        # Basic mode: use Bearer token
        token = self._get_access_token()
        return {"Authorization": f"Bearer {token}"}

    def _prepare_optional_headers(self, **kwargs: Any) -> dict[str, str]:
        """Prepare optional headers from kwargs.

        Args:
            **kwargs: Keyword arguments that may contain optional headers.

        Returns:
            Dictionary with prepared headers.
        """
        headers = {}

        # Map of parameter names to header names
        optional_headers = {
            "x_client_id": "X-Client-ID",
            "x_request_id": "X-Request-ID",
            "x_session_id": "X-Session-ID",
        }

        for key, header in optional_headers.items():
            if key in kwargs:
                value = kwargs.pop(key)
                if value:  # Only add if not None or empty
                    headers[header] = str(value)
                    logger.debug(f"Added header {header}: {headers[header]}")

        return headers

    def _get_access_token(self) -> str:
        """Get access token for Basic authentication mode.

        Note: This method is only called in Basic mode, not in mTLS mode.

        Returns:
            Valid access token.

        Raises:
            requests.RequestException: If token request fails.
            ValueError: If response format is invalid.
        """
        if self._is_token_valid() and self.access_token:
            logger.debug("Returning cached valid token")
            return self.access_token

        if not self.config.auth_basic:
            raise ValueError("Basic authentication is enabled but auth_basic is not set")

        logger.info("Requesting new access token using basic authentication")
        logger.debug(f"OAuth URL: {self.config.oauth_url}")

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {self.config.auth_basic}",
        }

        data = {"scope": self.config.scope}
        logger.debug(f"Token request data: scope={self.config.scope}")

        try:
            response = self.session.post(
                self.config.oauth_url,
                headers=headers,
                data=data,
                timeout=self.config.request_timeout,
            )
            response.raise_for_status()

            token_data: dict[str, Any] = response.json()
            access_token = token_data.get("access_token")
            if not access_token:
                raise ValueError("No access_token in response")

            self.access_token = str(access_token)
            logger.debug("Access token received successfully")

            # Safely parse expiration time
            self.token_expires_at = self._safe_parse_expiration(token_data)

            if self.token_expires_at:
                expiry_time = datetime.fromtimestamp(self.token_expires_at, tz=UTC)
                logger.info(f"Token obtained, expires at {expiry_time}")
                logger.debug(f"Token expiration timestamp: {self.token_expires_at}")
            else:
                logger.info("Token obtained with unknown expiration")

            return str(access_token)

        except requests.RequestException as e:
            logger.error(f"Failed to get access token: {e}")
            raise
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid token response format: {e}")
            raise ValueError(f"Invalid token response: {e}")

    def get_access_token(self, force_refresh: bool = False) -> str:
        """Get access token, refreshing if necessary.

        Note: In mTLS mode, this method returns empty string as no token is needed.

        Args:
            force_refresh: Force token refresh even if current token is valid.

        Returns:
            Valid access token in Basic mode, empty string in mTLS mode.

        Raises:
            requests.RequestException: If token request fails (Basic mode only).
            ValueError: If response format is invalid (Basic mode only).
        """
        if self.config.is_cert_auth():
            logger.debug("mTLS mode active, no access token needed")
            return ""

        logger.debug(f"get_access_token called with force_refresh={force_refresh}")

        if force_refresh:
            self.access_token = None
            self.token_expires_at = None

        return self._get_access_token()

    def get_models(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Get list of available models.

        Args:
            **kwargs: Optional parameters including:
                x_request_id: Request ID for tracing.
                x_session_id: Session ID for tracing.
                x_client_id: Client ID for tracing.

        Returns:
            List of model information dictionaries.

        Raises:
            requests.RequestException: If request fails.
        """
        logger.debug("get_models called")

        headers = {
            "Accept": "application/json",
            **self._get_auth_headers(),
            **self._prepare_optional_headers(**kwargs),
        }

        try:
            logger.debug(f"Requesting models from: {self.config.models_url}")
            response = self.session.get(
                self.config.models_url,
                headers=headers,
                timeout=self.config.request_timeout,
            )
            response.raise_for_status()

            models_data: dict[str, Any] = response.json()
            models: list[dict[str, Any]] = models_data.get("data", [])
            logger.debug(f"Retrieved {len(models)} models")
            return models

        except requests.RequestException as e:
            logger.error(f"Failed to get models: {e}")
            raise

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str = "GigaChat",
        temperature: float | None = None,
        top_p: float | None = None,
        stream: bool = False,
        max_tokens: int | None = None,
        repetition_penalty: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | Generator[dict[str, Any], None, None]:
        """Get chat completion from the model.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            model: Model ID to use.
            temperature: Sampling temperature (>0).
            top_p: Alternative to temperature, nucleus sampling (0-1).
            stream: Whether to stream the response.
            max_tokens: Maximum tokens in response.
            repetition_penalty: Penalty for repetitions.
            **kwargs: Additional parameters for the API including:
                x_request_id: Request ID for tracing.
                x_session_id: Session ID for tracing.
                x_client_id: Client ID for tracing.

        Returns:
            API response dictionary or generator for streaming.

        Raises:
            requests.RequestException: If request fails.
            ValueError: If parameters are invalid.
        """
        logger.debug(
            f"chat_completion called with model={model}, stream={stream}, "
            f"temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}, "
            f"repetition_penalty={repetition_penalty}, messages_count={len(messages)}"
        )

        if not messages:
            raise ValueError("Messages list cannot be empty")

        # Log first message content (truncated for security)
        if logger.isEnabledFor(logging.DEBUG) and messages:
            first_msg = messages[0]
            content_preview = first_msg.get("content", "")[:100]
            logger.debug(
                f"First message role: {first_msg.get('role')}, content preview: {content_preview}..."
            )

        # Prepare headers based on stream mode
        if stream:
            accept_header = "text/event-stream"
        else:
            accept_header = "application/json"

        headers = {
            "Content-Type": "application/json",
            "Accept": accept_header,
            **self._get_auth_headers(),
            **self._prepare_optional_headers(**kwargs),
        }

        data: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}

        # Add optional parameters
        if temperature is not None:
            data["temperature"] = temperature
        if top_p is not None:
            data["top_p"] = top_p
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if repetition_penalty is not None:
            data["repetition_penalty"] = repetition_penalty

        # Add only allowed additional parameters from kwargs
        allowed_params = {
            "functions",
            "function_call",
            "attachments",
            "update_interval",
            "n",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
        }

        for key, value in kwargs.items():
            if key in allowed_params:
                data[key] = value
                logger.debug(f"Added parameter {key}: {value}")

        try:
            logger.debug(f"Sending request to: {self.config.completion_url}")

            if stream:
                # For streaming responses with configurable timeout
                response = self.session.post(
                    self.config.completion_url,
                    headers=headers,
                    json=data,
                    timeout=(
                        self.config.request_timeout,
                        self.config.stream_timeout,
                    ),  # (connect, read) timeout
                    stream=True,
                )
                response.raise_for_status()
                logger.debug("Streaming response initiated")

                def stream_generator() -> Generator[dict[str, Any], None, None]:
                    """Generate streaming response chunks.

                    Yields:
                        Parsed JSON chunks from the SSE stream.
                    """
                    chunk_count = 0
                    try:
                        for line in response.iter_lines(decode_unicode=True):
                            if line:
                                if line.startswith("data: "):
                                    data_str = line[6:]  # Remove 'data: ' prefix
                                    if data_str == "[DONE]":
                                        logger.debug(f"Stream completed after {chunk_count} chunks")
                                        break
                                    try:
                                        chunk = json.loads(data_str)
                                        chunk_count += 1
                                        yield chunk
                                    except json.JSONDecodeError:
                                        logger.warning(
                                            f"Failed to parse streaming data: {data_str}"
                                        )
                    finally:
                        response.close()
                        logger.debug("Stream connection closed")

                return stream_generator()
            # For non-streaming responses
            response = self.session.post(
                self.config.completion_url,
                headers=headers,
                json=data,
                timeout=self.config.request_timeout,
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()

            if logger.isEnabledFor(logging.DEBUG):
                # Log response metrics
                if "usage" in result:
                    usage = result["usage"]
                    logger.debug(
                        f"Token usage - prompt: {usage.get('prompt_tokens')}, "
                        f"completion: {usage.get('completion_tokens')}, "
                        f"total: {usage.get('total_tokens')}"
                    )

            return result

        except requests.RequestException as e:
            logger.error(f"Failed to get chat completion: {e}")
            raise

    def create_embeddings(
        self,
        input_texts: str | list[str],
        model: str = "Embeddings",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create embeddings for input texts.

        Args:
            input_texts: Single text or list of texts to embed.
            model: Model to use ('Embeddings' or 'EmbeddingsGigaR').
            **kwargs: Optional parameters including:
                x_request_id: Request ID for tracing.
                x_session_id: Session ID for tracing.
                x_client_id: Client ID for tracing.

        Returns:
            API response with embeddings.

        Raises:
            requests.RequestException: If request fails.
            ValueError: If parameters are invalid.
        """
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        elif not input_texts:
            raise ValueError("Input texts cannot be empty")

        logger.debug(f"create_embeddings called with model={model}, input_count={len(input_texts)}")

        if model not in ["Embeddings", "EmbeddingsGigaR"]:
            raise ValueError(f"Invalid embeddings model: {model}")

        # Log text lengths for debugging
        if logger.isEnabledFor(logging.DEBUG):
            text_lengths = [len(text) for text in input_texts]
            logger.debug(f"Input text lengths: {text_lengths}")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self._get_auth_headers(),
            **self._prepare_optional_headers(**kwargs),
        }

        data = {"model": model, "input": input_texts}

        try:
            logger.debug(f"Sending embeddings request to: {self.config.embeddings_url}")
            response = self.session.post(
                self.config.embeddings_url,
                headers=headers,
                json=data,
                timeout=self.config.request_timeout,
            )
            response.raise_for_status()

            result: dict[str, Any] = response.json()

            if logger.isEnabledFor(logging.DEBUG):
                embeddings_data = result.get("data", [])
                logger.debug(f"Created {len(embeddings_data)} embeddings")
                if embeddings_data:
                    first_embedding_size = len(embeddings_data[0].get("embedding", []))
                    logger.debug(f"Embedding dimension: {first_embedding_size}")

            return result

        except requests.RequestException as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise

    def close(self) -> None:
        """Close the client session."""
        logger.debug("Closing GigaChatClient session")
        self.session.close()

    def __enter__(self) -> GigaChatClient:
        """Context manager entry."""
        logger.debug("Entering GigaChatClient context")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        logger.debug(f"Exiting GigaChatClient context - exc_type: {exc_type}, exc_val: {exc_val}")
        self.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # DEBUG
        format="[%(asctime)s][%(name)s][%(levelname)s][%(filename)s:%(lineno)d][%(message)s]",
    )
    # Test the client
    try:
        # Initialize client
        client = GigaChatClient()

        # Log authentication method being used
        auth_mode = "mTLS" if client.config.is_cert_auth() else "Basic"
        logger.info(f"Testing GigaChat client in {auth_mode} authentication mode")

        # Test 1: Get access token (only relevant for Basic mode)
        if not client.config.is_cert_auth():
            logger.info("=" * 60)
            logger.info("Test 1: Getting access token")
            token = client.get_access_token()
            logger.info(f"Access token obtained: {token[:20]}...")
        else:
            logger.info("=" * 60)
            logger.info("Test 1: Skipping token test (mTLS mode)")

        # Test 2: Get models with optional tracing headers
        logger.info("=" * 60)
        logger.info("Test 2: Getting models list")
        request_id = str(uuid.uuid4())
        models = client.get_models(x_request_id=request_id)
        logger.info(f"Found {len(models)} models (request_id: {request_id})")

        # Sort models alphabetically by ID for consistent output
        sorted_models = sorted(models, key=lambda m: m.get("id", "").lower())

        # Display all models in alphabetical order
        for model in sorted_models:
            logger.info(f"  - {model.get('id')}: {model.get('type')}")

        # Test 3: Chat completion (non-streaming) with tracing
        logger.info("=" * 60)
        logger.info("Test 3: Chat completion (non-streaming)")
        messages = [
            {
                "role": "user",
                "content": "Привет! Ответь в одном предложении: что ты умеешь?",
            }
        ]

        session_id = str(uuid.uuid4())
        response = client.chat_completion(messages, model="GigaChat", x_session_id=session_id)
        # Type guard for non-streaming response
        if isinstance(response, dict) and response.get("choices"):
            content = response["choices"][0]["message"]["content"]
            logger.info(f"Model response: {content}")
            logger.info(f"Session ID: {session_id}")

        # Test 4: Create embeddings with tracing
        logger.info("=" * 60)
        logger.info("Test 4: Creating embeddings")
        texts = ["Привет, мир!", "Тестовый текст для эмбеддингов"]

        try:
            request_id = str(uuid.uuid4())
            embeddings_response = client.create_embeddings(texts, x_request_id=request_id)
            embeddings_data = embeddings_response.get("data", [])
            logger.info(f"Created {len(embeddings_data)} embeddings (request_id: {request_id})")
            for i, emb in enumerate(embeddings_data):
                emb_vector = emb.get("embedding", [])
                logger.info(f"  - Text {i}: embedding size = {len(emb_vector)}")
        except requests.HTTPError as e:
            if e.response and e.response.status_code == 402:
                logger.warning(
                    "Embeddings API requires payment (402 Payment Required). "
                    "This is expected for some subscription tiers. Skipping embeddings test."
                )
            else:
                logger.error(f"Failed to create embeddings: {e}")
                raise

        # Test 5: Streaming (optional test)
        logger.info("=" * 60)
        logger.info("Test 5: Streaming response")
        messages = [
            {
                "role": "user",
                "content": "Write a short poem about programming in English.",
            }
        ]

        try:
            stream_response = client.chat_completion(
                messages,
                model="GigaChat",
                stream=True,
                x_session_id=session_id,  # Reuse session ID for continuity
            )
            logger.info(f"Streaming response (session_id: {session_id}):")
            # Type guard for streaming response
            if not isinstance(stream_response, dict):
                for chunk in stream_response:
                    if chunk.get("choices"):
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            print(delta["content"], end="", flush=True)
            print()  # New line after streaming
        except requests.HTTPError as e:
            if e.response and e.response.status_code == 402:
                logger.warning(
                    "Streaming API requires payment (402 Payment Required). "
                    "This is expected for some subscription tiers. Skipping streaming test."
                )
            else:
                logger.error(f"Failed to get streaming response: {e}")
                raise

        # Close client
        client.close()
        logger.info("\nAll tests completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import sys

        sys.exit(1)
