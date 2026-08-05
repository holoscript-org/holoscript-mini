"""
Pipeline progress-event contract.

Every stage in `pipeline/pipeline_runner.py::run_pipeline()` — and the stage
functions it calls into (prompt optimizer, intent extractor, the three scene
architect passes, the critic/fixer loop) — can report its progress by calling
an `on_event` callback with a `PipelineEvent`. This module defines that event
shape and nothing else: it has zero knowledge of WebSockets, asyncio, or
FastAPI. The transport bridge (turning these events into WebSocket messages)
lives entirely in `backend/api_server.py`.

`on_event` is always optional (defaults to `None` throughout the pipeline)
so every existing call site — `run_with_voice()`, the `python -m
pipeline.pipeline_runner` CLI entry point, and the pre-existing `POST /command`
handler in `backend/api_server.py` — keeps working unchanged, with no event
emission, if it doesn't pass one.

MIRROR STATUS: this dataclass's shape is mirrored in TypeScript at
`gui/lib/pipelineTypes.ts` (the wire format the frontend consumes over
`/ws/pipeline`). Any field added/removed/renamed here must be mirrored there
too — same convention as the `pipeline/scene_validator.py` /
`gui/lib/sceneFactory.ts` schema mirror.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

# Status values a stage event can carry.
STARTED = "started"
OUTPUT = "output"
COMPLETED = "completed"
FAILED = "failed"


@dataclass
class PipelineEvent:
    run_id: str
    stage: str              # machine id, e.g. "prompt_optimizer", "architect_layout", "critic_iteration_2"
    status: str              # "started" | "output" | "completed" | "failed"
    label: str               # human-readable stage name, e.g. "Prompt Optimizer"
    payload: dict[str, Any] | None = None
    timestamp: float = field(default_factory=time.time)
    elapsed_ms: int | None = None   # set on "completed"/"failed" only
    provider: str | None = None      # "gemini" | "groq" | None, when an LLM call was involved


OnEvent = Callable[[PipelineEvent], None]


def new_run_id() -> str:
    return uuid.uuid4().hex


def make_emitter(run_id: str, on_event: OnEvent | None) -> Callable[..., None]:
    """
    Build a small `emit(stage, status, label, payload=None, provider=None,
    elapsed_ms=None)` closure. No-ops (does nothing, costs nothing extra)
    when `on_event` is None, so call sites can unconditionally call
    `emit(...)` without checking whether anyone is listening.
    """
    def emit(
        stage: str,
        status: str,
        label: str,
        payload: dict[str, Any] | None = None,
        provider: str | None = None,
        elapsed_ms: int | None = None,
    ) -> None:
        if on_event is None:
            return
        on_event(
            PipelineEvent(
                run_id=run_id,
                stage=stage,
                status=status,
                label=label,
                payload=payload,
                provider=provider,
                elapsed_ms=elapsed_ms,
            )
        )

    return emit


def cli_event_printer(event: PipelineEvent) -> None:
    """
    Trivial `on_event` implementation for CLI/voice entry points — improved
    stage visibility on stdout, purely additive, no behavior change to the
    pipeline itself. Not used unless a caller opts in.
    """
    suffix = f" ({event.elapsed_ms}ms)" if event.elapsed_ms is not None else ""
    provider = f" via {event.provider}" if event.provider else ""
    print(f"[pipeline] {event.label}: {event.status}{provider}{suffix}")
