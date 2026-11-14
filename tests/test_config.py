"""
Configuration surface — no hardcoded credentials, projects, or model ids.

Two incidents motivate these tests:
  1. llm/gemini_client.py defaulted GCP_PROJECT to a shared university project,
     so a missing .env silently billed the wrong account.
  2. GEMINI_ARCHITECT_MODEL was declared in .env but read by nothing after the
     2026-08 restructure hardcoded the model ids — setting it did nothing.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from llm import gemini_client
from llm.gemini_client import model_flash, model_groq, model_pro, stage_model

_ROOT = Path(__file__).resolve().parents[1]


# ── no fallback GCP project ──────────────────────────────────────────────────

def test_no_vertex_client_without_project(monkeypatch):
    """Unset GCP_PROJECT must disable Vertex AI, never guess a project."""
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    monkeypatch.setattr(gemini_client, "_vertex_client", None)
    monkeypatch.setattr(gemini_client, "_vertex_client_failed", False)
    assert gemini_client.get_vertex_client() is None


def test_no_university_project_id_anywhere_in_source():
    """The old shared project id must not survive as a default or literal."""
    for path in list((_ROOT / "pipeline").glob("*.py")) + list((_ROOT / "llm").glob("*.py")):
        assert "reportevaluator" not in path.read_text(encoding="utf-8"), \
            f"{path.name} still references the old shared GCP project"


def test_missing_project_falls_back_to_groq(monkeypatch):
    """The pipeline must degrade to Groq, not crash, when Vertex is unavailable."""
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    monkeypatch.setattr(gemini_client, "_vertex_client", None)
    monkeypatch.setattr(gemini_client, "_vertex_client_failed", False)
    monkeypatch.setattr(gemini_client, "call_groq", lambda *a, **k: "groq-said-this")
    raw, provider = gemini_client.call_llm("pro", "groq", "prompt", "system")
    assert (raw, provider) == ("groq-said-this", "groq")


# ── model resolution ─────────────────────────────────────────────────────────

def test_tier_defaults(monkeypatch):
    for key in ["GEMINI_PRO_MODEL", "GEMINI_FLASH_MODEL", "GROQ_MODEL"]:
        monkeypatch.delenv(key, raising=False)
    assert model_pro() == "gemini-2.5-pro"
    assert model_flash() == "gemini-2.5-flash"
    assert model_groq() == "llama-3.3-70b-versatile"


def test_tier_overrides_apply(monkeypatch):
    monkeypatch.setenv("GEMINI_PRO_MODEL", "gemini-experimental")
    monkeypatch.setenv("GROQ_MODEL", "llama-tiny")
    assert model_pro() == "gemini-experimental"
    assert model_groq() == "llama-tiny"


def test_stage_override_beats_tier(monkeypatch):
    monkeypatch.setenv("GEMINI_FLASH_MODEL", "flash-default")
    monkeypatch.setenv("GEMINI_CRITIC_MODEL", "critic-special")
    assert stage_model("GEMINI_CRITIC_MODEL", "flash") == "critic-special"
    assert stage_model("GEMINI_VERIFIER_MODEL", "flash") == "flash-default"


def test_stage_falls_back_to_tier_when_unset(monkeypatch):
    monkeypatch.delenv("GEMINI_ARCHITECT_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_PRO_MODEL", "pro-x")
    assert stage_model("GEMINI_ARCHITECT_MODEL", "pro") == "pro-x"


def test_models_resolved_at_call_time_not_import_time(monkeypatch):
    """Changing the environment must take effect without re-importing."""
    monkeypatch.setenv("GEMINI_PRO_MODEL", "first")
    assert model_pro() == "first"
    monkeypatch.setenv("GEMINI_PRO_MODEL", "second")
    assert model_pro() == "second"


@pytest.mark.parametrize(
    "module,env_key",
    [
        ("scene_architect",  "GEMINI_ARCHITECT_MODEL"),
        ("scene_architect",  "GEMINI_FINISH_MODEL"),
        ("critic_agent",     "GEMINI_CRITIC_MODEL"),
        ("prompt_optimizer", "GEMINI_OPTIMIZER_MODEL"),
        ("intent_extractor", "GEMINI_INTENT_MODEL"),
        ("intent_verifier",  "GEMINI_VERIFIER_MODEL"),
    ],
)
def test_every_llm_stage_declares_its_override_key(module, env_key):
    """
    Guards the exact regression that killed GEMINI_ARCHITECT_MODEL: a stage
    quietly hardcoding its model so the documented env var does nothing.
    """
    src = (_ROOT / "pipeline" / f"{module}.py").read_text(encoding="utf-8")
    assert env_key in src, f"{module}.py no longer references {env_key}"
    assert "stage_model(" in src, f"{module}.py does not resolve its model via stage_model()"


def test_no_hardcoded_model_ids_outside_the_client():
    """Model literals may live only in llm/gemini_client.py's tier defaults."""
    pattern = re.compile(r'"(gemini-[\d.]+[a-z-]*|llama-[\d.]+[a-z0-9-]*)"')
    for path in (_ROOT / "pipeline").glob("*.py"):
        hits = pattern.findall(path.read_text(encoding="utf-8"))
        assert not hits, f"{path.name} hardcodes model id(s): {hits}"


# ── env template completeness ────────────────────────────────────────────────

def _declared_keys(path: Path) -> set[str]:
    return set(re.findall(r"^([A-Z][A-Z0-9_]*)=", path.read_text(encoding="utf-8"), re.M))


def test_env_example_exists_and_has_no_secrets():
    example = _ROOT / ".env.example"
    assert example.exists()
    text = example.read_text(encoding="utf-8")
    assert "gsk_" not in text, ".env.example contains a real Groq key"
    assert "mongodb+srv://" not in text or "MONGODB_URI=\n" in text, \
        ".env.example contains a real Mongo connection string"


def test_env_example_covers_every_key_used_in_code():
    documented = _declared_keys(_ROOT / ".env.example")
    commented = set(re.findall(
        r"^#\s*([A-Z][A-Z0-9_]*)=", (_ROOT / ".env.example").read_text(encoding="utf-8"), re.M))
    documented |= commented

    used: set[str] = set()
    for folder in ("pipeline", "llm", "backend"):
        for path in (_ROOT / folder).rglob("*.py"):
            used |= set(re.findall(r'getenv\(\s*"([A-Z][A-Z0-9_]*)"', path.read_text(encoding="utf-8")))

    # POLY_PIZZA_API_KEY is a legacy alias checked alongside POLYPIZZA_API_KEY
    used.discard("POLY_PIZZA_API_KEY")

    missing = used - documented
    assert not missing, f"env vars read in code but absent from .env.example: {sorted(missing)}"


def test_frontend_env_example_exists():
    example = _ROOT / "gui" / ".env.local.example"
    assert example.exists()
    assert "NEXT_PUBLIC_API_URL" in example.read_text(encoding="utf-8")


def test_no_hardcoded_backend_origin_in_frontend():
    """lib/api.ts owns the origin; no other module may hardcode it."""
    offenders = []
    for folder in ("components", "hooks", "app"):
        for path in (_ROOT / "gui" / folder).rglob("*.ts*"):
            text = path.read_text(encoding="utf-8")
            if re.search(r'["\']https?://(localhost|127\.0\.0\.1):\d+', text):
                offenders.append(path.name)
    assert not offenders, f"hardcoded backend origin in: {offenders}"
