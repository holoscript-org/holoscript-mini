/**
 * MIRROR STATUS: this file mirrors the wire format produced by
 * `pipeline/events.py`'s `PipelineEvent` dataclass and serialized by
 * `backend/api_server.py`'s `/ws/pipeline` route (`_event_to_wire`). Any
 * field added/removed/renamed on the Python side must be mirrored here too
 * — same convention as `pipeline/scene_validator.py` / `lib/sceneFactory.ts`.
 *
 * This is the SINGLE SOURCE OF TRUTH on the TypeScript side for the pipeline
 * progress-event wire protocol consumed by hooks/usePipelineStream.ts.
 */

export type PipelineEventStatus = "started" | "output" | "completed" | "failed"

export interface PipelineEvent {
  run_id: string
  stage: string // machine id, e.g. "prompt_optimizer", "architect_layout", "critic_iteration_2"
  status: PipelineEventStatus
  label: string // human-readable stage name, e.g. "Prompt Optimizer"
  payload: Record<string, unknown> | null
  timestamp: number
  elapsed_ms: number | null
  provider: "gemini" | "groq" | null
}

export interface WirePipelineEventMessage extends PipelineEvent {
  type: "pipeline_event"
}

export interface WireRunStartedMessage {
  type: "run_started"
  run_id: string
  transcript: string
}

export interface WireRunFinishedMessage {
  type: "run_finished"
  run_id: string | null
  status: "done" | "error"
  scene?: Record<string, unknown>
  message?: string
}

export type WireMessage = WirePipelineEventMessage | WireRunStartedMessage | WireRunFinishedMessage

// ─── Per-stage payload shapes (see pipeline/*.py docstrings for the ─────────
// ─── authoritative Python-side contract each of these mirrors) ─────────────

export interface PromptOptimizerPayload {
  optimized_prompt: string
  clarifications_made: string[]
  original_prompt: string
}

export interface IntentObjectEntry {
  concept: string
  count: number
  role: "primary" | "secondary" | "detail"
  notes: string
}

export interface SpatialRelationship {
  subject: string
  relation: string
  object: string
}

export interface DynamicsEntry {
  target: string
  motion: string
  notes: string
}

export interface SceneIntentIR {
  scene_type: string
  objects: IntentObjectEntry[]
  spatial_relationships: SpatialRelationship[]
  dynamics: DynamicsEntry[]
  mood_style: { lighting_mood?: string; descriptors?: string[] }
  educational_focus: string
  explicit_user_constraints: string[]
}

export interface VerifiedAsset {
  concept: string
  path: string
  label: string
  score: string
}

export interface AssetResolutionPayload {
  verified_assets?: VerifiedAsset[]
  candidates?: { concept: string; category: string }[]
  downloaded?: VerifiedAsset[]
  objects?: string[]
  structures?: string[]
  systems?: string[]
  effects?: string[]
  resolved?: Record<string, string[]>
  unresolved?: string[]
}

export interface ArchitectLayoutObject {
  id: string
  type: "primitive" | "mesh"
  geometry?: { type: string }
  model?: string
  position: [number, number, number]
  parent?: string
  label?: string
}

export interface ArchitectLayoutPayload {
  objects: ArchitectLayoutObject[]
}

export interface ArchitectDetailPayload {
  objects: Array<Record<string, unknown> & { id: string; material?: { color?: string }; geometry?: { type?: string }; animation?: { type?: string; physics_type?: string }; description?: string }>
}

export interface ArchitectFinishPayload {
  name: string
  summary: string
  lights: Array<{ type: string; intensity: number; color?: string; position?: [number, number, number] }>
  camera: { position: [number, number, number]; target: [number, number, number]; fov?: number }
}

export interface CriticIssue {
  category: string
  objects: string[]
  description: string
  fix: string
}

export interface CriticIterationPayload {
  iteration: number
  verdict?: "OK" | "HAS_ISSUES"
  issues?: CriticIssue[]
  fixed?: boolean
  object_count_before?: number
  object_count_after?: number
  issues_addressed?: CriticIssue[]
}

export interface ValidatePayload {
  fatal: string | null
  errors: string[]
}

export interface RepairPayload {
  actions: string[]
}
