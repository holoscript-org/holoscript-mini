import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "HYBRID").upper()

VALID_PROVIDERS = ("GROQ", "OLLAMA", "HYBRID")
if PROVIDER not in VALID_PROVIDERS:
    raise ValueError(f"LLM_PROVIDER must be one of {VALID_PROVIDERS}, got {PROVIDER}")


def get_provider() -> str:
    return PROVIDER
