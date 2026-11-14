"""
Shared test setup.

These tests are deliberately hermetic: no network, no MongoDB, no Redis, no LLM
calls. Anything that would reach outside the process is mocked at its seam. That
keeps the suite runnable on a laptop with no .env, which is the only way it will
actually get run before a demo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _ROOT


@pytest.fixture
def minimal_object() -> dict:
    """Smallest object that passes validation — build variants from this."""
    return {
        "id": "obj_a",
        "type": "primitive",
        "geometry": {"type": "sphere", "radius": 1.0},
        "position": [0, 0, 0],
        "material": {
            "type": "standard",
            "color": "#ff8800",
            "roughness": 0.5,
            "metalness": 0.0,
        },
        "animation": {"type": "none"},
    }


@pytest.fixture
def minimal_scene(minimal_object: dict) -> dict:
    return {
        "name": "Test Scene",
        "objects": [minimal_object],
        "lights": [{"type": "ambient", "intensity": 0.4, "color": "#ffffff"}],
        "camera": {"position": [0, 5, 20], "target": [0, 0, 0], "fov": 60},
    }


@pytest.fixture
def make_concept_doc():
    """
    Factory fixture building a MongoDB concept document for kb_cleanup tests.

    `live_search=True` sets the "Downloaded via Poly Pizza" description marker
    that kb_cleanup uses as its provenance check — the sole thing protecting
    seed concepts from deletion.
    """
    def _make(
        concept_id: str,
        asset_src: str = "",
        live_search: bool = True,
        doc_type: str = "object",
    ) -> dict:
        description = (
            f"Downloaded via Poly Pizza: {concept_id}"
            if live_search
            else f"{concept_id} seeded concept"
        )
        return {
            "_id": concept_id,
            "asset_src": asset_src,
            "category": "abstract",
            "type": doc_type,
            "description": description,
            "synonyms": [concept_id],
        }

    return _make
