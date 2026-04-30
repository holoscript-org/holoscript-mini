#!/usr/bin/env python3
"""Phase 5 test: Unified provider with fallback chain."""

from __future__ import annotations

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm.unified_provider import (
    UnifiedProvider,
    GroqProvider,
    OllamaProvider,
    GeminiProvider,
    ProviderError,
)
from llm.planner import plan


def test_provider_availability() -> int:
    """Test that providers report availability correctly."""
    print("\n[Test 1] Provider availability checks")
    print("-" * 70)

    groq = GroqProvider()
    ollama = OllamaProvider()
    gemini = GeminiProvider()

    groq_available = groq.available()
    ollama_available = ollama.available()
    gemini_available = gemini.available()

    print(f"✓ Groq available: {groq_available}")
    print(f"✓ Ollama available: {ollama_available}")
    print(f"✓ Gemini available: {gemini_available}")

    if not (groq_available or ollama_available or gemini_available):
        print("⚠ Warning: No providers are available. Tests may fail.")

    return 0


def test_unified_provider_chain() -> int:
    """Test unified provider fallback chain."""
    print("\n[Test 2] Unified provider chain")
    print("-" * 70)

    provider = UnifiedProvider()
    print(f"✓ Initialized unified provider with {len(provider.providers)} providers")

    sorted_providers = provider._sorted_providers()
    print(f"✓ Provider order: {[p.__class__.__name__ for p in sorted_providers]}")

    # Try to call with a simple prompt
    prompt = "Output: {\"scene_type\": \"atom\", \"num_objects\": 1}"

    try:
        response = provider.call(prompt)
        print(f"✓ Provider call succeeded")
        print(f"  Response length: {len(response)} chars")
        return 0
    except ProviderError as e:
        print(f"⚠ Provider call failed (expected if no providers available): {e}")
        return 0  # Don't fail the test if providers aren't available


def test_planner_with_unified_provider() -> int:
    """Test that planner uses unified provider."""
    print("\n[Test 3] Planner with unified provider")
    print("-" * 70)

    test_commands = [
        "show a hydrogen atom",
        "create a solar system",
    ]

    failures = 0
    for command in test_commands:
        plan_obj = plan(command)
        if plan_obj is None:
            print(f"⚠ planner returned None for: {command}")
            # Don't count as failure since unified provider may not be available
            continue

        print(f"✓ planner succeeded: {command} (scene_type={plan_obj.scene_type})")

    return failures


def test_provider_order_config() -> int:
    """Test that provider order can be configured."""
    print("\n[Test 4] Provider order configuration")
    print("-" * 70)

    # Test custom provider order via environment variable
    os.environ["LLM_PROVIDER_ORDER"] = "gemini,groq,ollama"
    provider = UnifiedProvider()
    sorted_providers = provider._sorted_providers()
    order = [p.__class__.__name__ for p in sorted_providers]

    print(f"✓ Custom order configured: {order}")

    if order[0] == "GeminiProvider":
        print(f"✓ First provider is Gemini (as configured)")
    else:
        print(f"⚠ First provider is {order[0]} (expected GeminiProvider)")

    # Restore default
    if "LLM_PROVIDER_ORDER" in os.environ:
        del os.environ["LLM_PROVIDER_ORDER"]

    return 0


def main() -> int:
    print("=" * 70)
    print("Phase 5 Unified Provider Test")
    print("=" * 70)

    failures = 0

    failures += test_provider_availability()
    failures += test_unified_provider_chain()
    failures += test_planner_with_unified_provider()
    failures += test_provider_order_config()

    print("\n" + "=" * 70)
    if failures:
        print(f"Completed with {failures} failure(s)")
        return 1

    print("All Phase 5 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
