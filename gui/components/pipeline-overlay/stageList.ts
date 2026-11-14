/**
 * Ordered list of pipeline stage ids + human labels shown in StageTimeline.
 * Mirrors pipeline/pipeline_runner.py's stage sequence exactly (see that
 * file's module docstring for the authoritative 11-stage list). Grouped ids
 * (e.g. all of semantic_parse/resolve_intent/asset_registry/live_search)
 * collapse into one AssetResolutionView panel since they're fast/mechanical,
 * but each still gets its own timeline row so progress stays granular.
 */

export interface StageDef {
  id: string
  label: string
  group: "optimizer" | "extraction" | "resolution" | "architect" | "critic" | "finalize"
}

export const STAGE_LIST: StageDef[] = [
  { id: "receive_transcript", label: "Receive Transcript", group: "optimizer" },
  { id: "prompt_optimizer", label: "Prompt Optimizer", group: "optimizer" },
  { id: "intent_extraction", label: "Intent Extraction", group: "extraction" },
  { id: "semantic_parse", label: "Semantic Parse", group: "resolution" },
  { id: "resolve_intent", label: "Resolve Intent", group: "resolution" },
  { id: "asset_registry", label: "Asset Registry", group: "resolution" },
  { id: "live_search", label: "Live Asset Search", group: "resolution" },
  { id: "architect_layout", label: "Layout & Composition", group: "architect" },
  { id: "architect_detail", label: "Object Detail", group: "architect" },
  { id: "architect_finish", label: "Lighting, Camera & Polish", group: "architect" },
  { id: "critic_iteration_1", label: "Critic Iteration 1", group: "critic" },
  { id: "critic_iteration_2", label: "Critic Iteration 2", group: "critic" },
  { id: "critic_iteration_3", label: "Critic Iteration 3", group: "critic" },
  { id: "validate", label: "Validate", group: "finalize" },
  { id: "repair", label: "Repair", group: "finalize" },
]

// Stages that only fire conditionally (legacy_fallback, and critic iterations
// beyond the first one) are not in the fixed list above — the timeline
// renders any stage id present in state.stageByName that isn't in this list
// as an extra trailing row, so nothing silently disappears.
