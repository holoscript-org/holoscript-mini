"""llm/unified_provider.py

Unified provider interface with automatic fallback chaining.
Supports Groq (primary), Ollama (secondary), and Gemini (tertiary) with
graceful degradation when providers are unavailable.
"""

from __future__ import annotations

import os
import json
import time
from typing import Optional, Callable
from dotenv import load_dotenv

from core.utils.logger import get_logger

load_dotenv()

logger = get_logger("unified_provider")


class ProviderError(Exception):
    """Raised when all providers fail."""

    pass


class GroqProvider:
    """Groq API provider."""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = "llama-3.1-8b-instant"
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.enabled = bool(self.api_key)

    def available(self) -> bool:
        return self.enabled

    def call(self, prompt: str) -> str:
        """Call Groq API."""
        if not self.enabled:
            raise ProviderError("Groq API key not configured")

        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a scene planner. Output ONLY valid JSON. No markdown. No explanations.",
                    },
                    {"role": "user", "content": prompt},
                ],
            }

            response = requests.post(self.url, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("Groq call failed: %s", e)
            raise ProviderError(f"Groq failed: {e}")


class OllamaProvider:
    """Ollama local provider."""

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "llama2")
        self.url = f"{self.base_url}/api/generate"
        self.enabled = True  # Always enabled; check availability at runtime

    def available(self) -> bool:
        """Check if Ollama is running."""
        try:
            import requests

            requests.get(f"{self.base_url}/api/tags", timeout=2)
            return True
        except Exception:
            return False

    def call(self, prompt: str) -> str:
        """Call Ollama API."""
        if not self.available():
            raise ProviderError("Ollama is not running")

        try:
            import requests

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }

            response = requests.post(self.url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            logger.error("Ollama call failed: %s", e)
            raise ProviderError(f"Ollama failed: {e}")


class GeminiProvider:
    """Google Gemini API provider."""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = "gemini-1.5-flash"
        self.enabled = bool(self.api_key)

    def available(self) -> bool:
        return self.enabled

    def call(self, prompt: str) -> str:
        """Call Gemini API."""
        if not self.enabled:
            raise ProviderError("Gemini API key not configured")

        try:
            import requests

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            headers = {
                "Content-Type": "application/json",
            }
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt,
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                },
            }

            params = {"key": self.api_key}
            response = requests.post(url, headers=headers, json=payload, params=params, timeout=30)
            response.raise_for_status()
            result = response.json()
            # Extract text from Gemini response
            candidates = result.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            raise ProviderError("No response content from Gemini")
        except Exception as e:
            logger.error("Gemini call failed: %s", e)
            raise ProviderError(f"Gemini failed: {e}")


def _extract_json_from_text(text: str) -> dict | None:
    """Extract JSON from LLM response."""
    try:
        start_idx = text.find("{")
        if start_idx == -1:
            return None
        end_idx = text.rfind("}") + 1
        if end_idx <= start_idx:
            return None
        json_str = text[start_idx:end_idx]
        return json.loads(json_str)
    except Exception:
        return None


class UnifiedProvider:
    """Unified provider with automatic fallback chain."""

    def __init__(self):
        self.providers = [
            GroqProvider(),
            OllamaProvider(),
            GeminiProvider(),
        ]
        self.call_order = os.getenv("LLM_PROVIDER_ORDER", "groq,ollama,gemini").lower().split(",")
        logger.info("unified_provider: chain order = %s", self.call_order)

    def _sorted_providers(self) -> list:
        """Return providers in configured order."""
        priority_map = {name.strip(): i for i, name in enumerate(self.call_order)}

        def priority(provider):
            if isinstance(provider, GroqProvider):
                return priority_map.get("groq", 999)
            elif isinstance(provider, OllamaProvider):
                return priority_map.get("ollama", 999)
            elif isinstance(provider, GeminiProvider):
                return priority_map.get("gemini", 999)
            return 999

        return sorted(self.providers, key=priority)

    def call(self, prompt: str) -> str:
        """Call providers in order until one succeeds."""
        sorted_providers = self._sorted_providers()
        errors = []

        for provider in sorted_providers:
            provider_name = provider.__class__.__name__
            if not provider.available():
                logger.debug("%s not available, skipping", provider_name)
                continue

            try:
                start = time.perf_counter()
                response = provider.call(prompt)
                elapsed = time.perf_counter() - start
                logger.info("%s succeeded (%.1fms)", provider_name, elapsed * 1000)
                return response
            except ProviderError as e:
                logger.warning("%s failed: %s", provider_name, e)
                errors.append(str(e))
                continue

        # All providers failed
        msg = f"All providers exhausted: {'; '.join(errors)}"
        logger.error(msg)
        raise ProviderError(msg)

    def call_and_extract_json(self, prompt: str) -> dict:
        """Call provider and extract JSON from response."""
        response = self.call(prompt)
        json_obj = _extract_json_from_text(response)
        if json_obj is None:
            raise ProviderError("Failed to extract JSON from response")
        return json_obj


# Singleton instance
_unified_provider: Optional[UnifiedProvider] = None


def get_unified_provider() -> UnifiedProvider:
    """Get or create the unified provider instance."""
    global _unified_provider
    if _unified_provider is None:
        _unified_provider = UnifiedProvider()
    return _unified_provider


__all__ = ["UnifiedProvider", "get_unified_provider", "ProviderError"]
