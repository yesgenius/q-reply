"""Configuration module for GigaChat API client.

This module provides a simple and reliable way to load configuration
from environment variables or .env file.

Usage:
    python -m gigachat.config

"""

import os
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv


class GigaChatConfig:
    """Simple configuration class for GigaChat API client.

    Loads configuration from environment variables with GIGACHAT_ prefix.
    Supports two authentication methods:
    - Basic authentication using auth_basic
    - Certificate-based authentication using TLS certificates

    At least one authentication method must be configured.

    Attributes:
        auth_basic: Base64-encoded authentication string.
        scope: API access scope (default: GIGACHAT_API_PERS).
        tls_cert: Path to TLS certificate file.
        tls_key: Path to TLS key file.
        tls_ca_cert: Path to CA certificate file.
        oauth_url: OAuth token endpoint URL.
        completion_url: Chat completions endpoint URL.
        models_url: Models listing endpoint URL.
        embeddings_url: Embeddings endpoint URL.
        request_timeout: Request timeout in seconds (default: 30).
        stream_timeout: Streaming read timeout in seconds (default: 300).
        max_retries: Maximum retry attempts (default: 3).
    """

    def __init__(self, env_file: Optional[str] = ".env"):
        """Initialize configuration from environment variables.

        Args:
            env_file: Path to .env file. If None, only system env vars are used.

        Raises:
            ValueError: If no authentication method is configured or if
                configuration is invalid.
        """
        # Load .env file if it exists
        if env_file and Path(env_file).exists():
            load_dotenv(env_file)

        # Load authentication settings
        self.auth_basic = os.getenv("GIGACHAT_AUTH_BASIC", "").strip() or None
        self.tls_cert = self._get_path("GIGACHAT_TLS_CERT")
        self.tls_key = self._get_path("GIGACHAT_TLS_KEY")
        self.tls_ca_cert = self._get_path("GIGACHAT_TLS_CA_CERT")

        # Validate authentication
        self._validate_auth()

        # Load API settings with defaults
        self.scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self.oauth_url = os.getenv(
            "GIGACHAT_OAUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        )
        self.completion_url = os.getenv(
            "GIGACHAT_COMPLETION_URL",
            "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
        )
        self.models_url = os.getenv(
            "GIGACHAT_MODELS_URL", "https://gigachat.devices.sberbank.ru/api/v1/models"
        )
        self.embeddings_url = os.getenv(
            "GIGACHAT_EMBEDDINGS_URL",
            "https://gigachat.devices.sberbank.ru/api/v1/embeddings",
        )

        # Load timeout settings with validation
        self.request_timeout = self._get_int("GIGACHAT_REQUEST_TIMEOUT", 30, 1, 300)
        self.stream_timeout = self._get_int("GIGACHAT_STREAM_TIMEOUT", 300, 30, 3600)
        self.max_retries = self._get_int("GIGACHAT_MAX_RETRIES", 3, 0, 10)

        # Validate scope
        self._validate_scope()

    def _get_path(self, env_var: str) -> Optional[Path]:
        """Get path from environment variable.

        Args:
            env_var: Environment variable name.

        Returns:
            Path object if env var is set and file exists, None otherwise.
        """
        value = os.getenv(env_var, "").strip()
        if not value:
            return None

        path = Path(value)
        if not path.exists():
            return None

        return path

    def _get_int(self, env_var: str, default: int, min_val: int, max_val: int) -> int:
        """Get integer value from environment variable with validation.

        Args:
            env_var: Environment variable name.
            default: Default value if not set.
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.

        Returns:
            Validated integer value.
        """
        value = os.getenv(env_var, str(default))
        try:
            result = int(value)
            if min_val <= result <= max_val:
                return result
        except ValueError:
            pass

        return default

    def _validate_auth(self) -> None:
        """Validate that at least one authentication method is configured.

        Raises:
            ValueError: If no valid authentication method is configured.
        """
        has_basic = bool(self.auth_basic)
        has_certs = bool(self.tls_cert and self.tls_key)

        if not has_basic and not has_certs:
            raise ValueError(
                "No authentication method configured. "
                "Set GIGACHAT_AUTH_BASIC or provide both "
                "GIGACHAT_TLS_CERT and GIGACHAT_TLS_KEY."
            )

        # If cert auth is partially configured, raise error
        if bool(self.tls_cert) != bool(self.tls_key):
            raise ValueError(
                "Both GIGACHAT_TLS_CERT and GIGACHAT_TLS_KEY must be provided "
                "for certificate authentication."
            )

        # Validate cert files exist if configured
        if has_certs and self.tls_cert and self.tls_key:
            if not self.tls_cert.is_file():
                raise ValueError(f"Certificate file not found: {self.tls_cert}")
            if not self.tls_key.is_file():
                raise ValueError(f"Key file not found: {self.tls_key}")

        # Validate CA cert if provided
        if self.tls_ca_cert and not self.tls_ca_cert.is_file():
            raise ValueError(f"CA certificate file not found: {self.tls_ca_cert}")

    def _validate_scope(self) -> None:
        """Validate API scope value.

        Raises:
            ValueError: If scope is not valid.
        """
        valid_scopes = {"GIGACHAT_API_PERS", "GIGACHAT_API_B2B", "GIGACHAT_API_CORP"}
        if self.scope not in valid_scopes:
            raise ValueError(
                f"Invalid scope: {self.scope}. "
                f"Must be one of: {', '.join(sorted(valid_scopes))}"
            )

    def is_basic_auth(self) -> bool:
        """Check if basic authentication is configured.

        Returns:
            True if basic auth is available, False otherwise.
        """
        return bool(self.auth_basic)

    def is_cert_auth(self) -> bool:
        """Check if certificate authentication is configured.

        Returns:
            True if certificate auth is available, False otherwise.
        """
        return bool(self.tls_cert and self.tls_key)

    def get_cert_paths(self) -> Optional[Tuple[str, str]]:
        """Get certificate and key paths for requests library.

        Returns:
            Tuple of (cert_path, key_path) if configured, None otherwise.
        """
        if self.tls_cert and self.tls_key:
            return (str(self.tls_cert), str(self.tls_key))
        return None

    def get_verify_path(self) -> Optional[str]:
        """Get CA certificate path for SSL verification.

        Returns:
            CA certificate path if configured, None otherwise.
        """
        if self.tls_ca_cert:
            return str(self.tls_ca_cert)
        return None


def load_config(env_file: Optional[str] = ".env") -> GigaChatConfig:
    """Load configuration from environment variables.

    Args:
        env_file: Path to .env file. If None, only system env vars are used.

    Returns:
        Loaded configuration instance.

    Raises:
        ValueError: If configuration is invalid.
    """
    return GigaChatConfig(env_file)


# Example usage and testing
if __name__ == "__main__":
    import sys

    try:
        config = load_config()

        print("Configuration loaded successfully!")
        print(
            f"Authentication method: {'Basic' if config.is_basic_auth() else 'Certificate'}"
        )
        print(f"Scope: {config.scope}")
        print(f"OAuth URL: {config.oauth_url}")
        print(f"Request timeout: {config.request_timeout}s")
        print(f"Stream timeout: {config.stream_timeout}s")
        print(f"Max retries: {config.max_retries}")

        if config.is_cert_auth():
            print(f"Certificate: {config.tls_cert}")
            print(f"Key: {config.tls_key}")
            if config.tls_ca_cert:
                print(f"CA Certificate: {config.tls_ca_cert}")

    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
