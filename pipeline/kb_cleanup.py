"""
Knowledge-base cleanup — remove junk concept documents created by the
phrase-gluing bug in live asset search.

Background
----------
`pipeline/live_search.py::_remember_asset_query()` uses the *raw search query*
as the MongoDB `_id`. Before the text_utils fix, `_extract_candidate_phrases()`
built n-grams from a stopword-stripped token list, so words that were never
adjacent got glued into nonsense phrases ("saturn and its rings" ->
"saturn its rings"). Every one of those phrases that returned any Poly Pizza
result became a permanent concept document pointing at an unrelated model:

    'its rings'                 -> /assets/meshes/abstract/onion_rings.glb
    'falling hardwood'          -> /assets/meshes/abstract/maple_trees.glb
    'harmonic motion'           -> /assets/meshes/abstract/harp.glb
    'google-style technical office' -> /assets/meshes/abstract/office_chair.glb

The 2026-08 pipeline restructure made this worse: live search now runs on the
Stage-2 *optimized* prompt (fluent LLM prose) rather than the terse transcript,
producing roughly 3x more garbage phrases per request.

Safety
------
- **Dry-run by default.** Pass --apply to actually delete.
- Only ever considers documents whose description marks them as live-search
  registrations ("Downloaded via Poly Pizza: ..."). The ~216 seed concepts from
  `asset_ingester.py` are never candidates, which protects legitimate
  multi-word entries like "human figure", "palm tree", "police car".
- Writes a full JSON backup of every document it deletes before deleting
  anything. MongoDB Atlas M0 (free tier) has no automated backups, so this file
  is the only way back.
- **Never touches GLB files on disk.** All 249 are tracked in git and may be
  referenced by other concepts; an orphaned file is harmless.

Usage
-----
    python -m pipeline.kb_cleanup              # dry run, prints the plan
    python -m pipeline.kb_cleanup --apply      # perform deletions
    python -m pipeline.kb_cleanup --restore backup.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from pipeline.knowledge_base.mongo_client import get_collection
from pipeline.text_utils import SEARCH_STOP, is_primitive_only, tokenize

_LIVE_SEARCH_MARKER = "Downloaded via Poly Pizza"
_BACKUP_DIR = _ROOT / "core" / "outputs" / "kb_backups"

# Category names and generic filler that asset_ingester.py registered as if they
# were concepts. These are SEED documents, so the provenance check that protects
# every other seed also protects these — hence a separate bucket behind its own
# --include-generic flag rather than folding them into the fragment rules.
#
# They are harmful because they match almost any prompt while pointing at one
# arbitrary member of their category: a scene mentioning "abstract" pulled an
# avocado, and "trees" resolves to a bed. Every one is redundant with a better
# singular that already exists ("tree", "building", "vehicle", "satellite",
# "person"), so deleting them loses no reachable asset.
_GENERIC_JUNK_IDS: dict[str, str] = {
    "abstract":     "category name; -> Avocado.glb, matched any 'abstract' phrasing",
    "buildings":    "category name; 'building' already exists",
    "humans":       "category name; 'person'/'human figure' already exist",
    "planets":      "category name; 'planet' + every named planet already exist",
    "satellites":   "category name; 'satellite' already exists",
    "trees":        "category name; -> bed.glb, and 'tree' already exists",
    "vehicles":     "category name; 'vehicle'/'car' already exist",
    "asset":        "ingester filler; -> castle_kit.glb",
    "asset pack":   "ingester filler; -> castle_kit.glb",
    "pack":         "ingester filler; -> castle_kit.glb",
    "primitives":   "ingester filler; -> skyscraper.glb, contradicts the primitive/mesh rule",
    "template":     "ingester filler; -> animated_character_base.glb",
}

# Human-reviewed junk that no word-list heuristic can catch: every word in these
# ids is a legitimate English noun, so they look structurally fine. They are
# wrong because the *mapping* is wrong, which only a human can judge. Each entry
# is annotated with why. Still restricted to live-search docs by classify().
_MANUAL_JUNK_IDS: dict[str, str] = {
    # concept id                    reason
    "harmonic motion":              "-> harp.glb; 'harmonic' matched 'harp' by string similarity",
    "icon":                         "-> heart.glb; meaningless key, matches any icon-like model",
    "grandfather clock pendulum":   "-> analog_clock.glb; three concepts glued, duplicates 'clock'",
    "hardwood court":               "-> floor.glb; generic floor keyed to a basketball-specific phrase",
    "loyal companion":              "-> german_shepard.glb; descriptive phrase, not a searchable concept",
    "house garden containing":      "-> houseplant.glb; fragment (already removed in first pass)",
}


def _is_live_search(doc: dict) -> bool:
    return str(doc.get("description", "")).startswith(_LIVE_SEARCH_MARKER)


def _asset_exists(doc: dict) -> bool:
    src = doc.get("asset_src") or ""
    return bool(src) and (_ROOT / "core" / str(src).lstrip("/")).exists()


def _fragment_words(concept_id: str) -> list[str]:
    """Words in the id that mark it as a sentence fragment rather than a thing."""
    return [w for w in tokenize(concept_id) if w in SEARCH_STOP]


def classify(docs: list[dict]) -> dict[str, list[dict]]:
    """
    Sort every document into exactly one bucket. Only live-search documents are
    ever eligible for deletion.
    """
    buckets: dict[str, list[dict]] = {
        "fragment": [],       # live-search id containing stopword/filler/motion words
        "manual": [],         # human-reviewed wrong mappings (see _MANUAL_JUNK_IDS)
        "generic": [],        # category names / ingester filler (see _GENERIC_JUNK_IDS)
        "primitive_only": [], # live-search id for a concept that must be primitives
        "broken_link": [],    # asset_src set but the file is gone
        "keep": [],
    }

    for doc in docs:
        cid = str(doc.get("_id", ""))
        live = _is_live_search(doc)

        # A missing file is a defect regardless of provenance, but only report
        # it when a path was actually claimed — effect/system/structure concepts
        # legitimately carry an empty asset_src.
        if doc.get("asset_src") and not _asset_exists(doc):
            buckets["broken_link"].append(doc)
            continue

        # Checked before the provenance gate: these are seeds by definition,
        # so a live-search-only rule can never reach them.
        if cid in _GENERIC_JUNK_IDS:
            buckets["generic"].append(doc)
            continue

        if live and cid in _MANUAL_JUNK_IDS:
            buckets["manual"].append(doc)
            continue

        if live and _fragment_words(cid):
            buckets["fragment"].append(doc)
            continue

        if live and is_primitive_only(cid):
            buckets["primitive_only"].append(doc)
            continue

        buckets["keep"].append(doc)

    return buckets


def _print_bucket(name: str, docs: list[dict], reason: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"{name}  ({len(docs)} documents)")
    print(f"  reason: {reason}")
    print("=" * 78)
    if not docs:
        print("  (none)")
        return
    for doc in sorted(docs, key=lambda d: str(d.get("_id", ""))):
        cid = str(doc.get("_id", ""))
        src = doc.get("asset_src") or "(none)"
        frags = _fragment_words(cid)
        note = f"   [fragment words: {', '.join(frags)}]" if frags else ""
        print(f"  {cid!r:36} -> {src}{note}")


def backup(docs: list[dict]) -> Path:
    _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = _BACKUP_DIR / f"kb_deleted_{stamp}.json"
    path.write_text(json.dumps(docs, indent=2, default=str), encoding="utf-8")
    return path


def restore(path: Path) -> int:
    docs = json.loads(path.read_text(encoding="utf-8"))
    coll = get_collection()
    restored = 0
    for doc in docs:
        coll.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        restored += 1
    return restored


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean junk concepts from the knowledge base.")
    ap.add_argument("--apply", action="store_true", help="actually delete (default is dry run)")
    ap.add_argument("--include-generic", action="store_true",
                    help="also delete category-name / filler seed concepts")
    ap.add_argument("--include-primitive-only", action="store_true",
                    help="also delete live-search concepts that should be primitives")
    ap.add_argument("--restore", metavar="BACKUP.json", help="re-insert documents from a backup file")
    args = ap.parse_args()

    coll = get_collection()

    if args.restore:
        n = restore(Path(args.restore))
        print(f"Restored {n} documents from {args.restore}")
        return

    docs = list(coll.find({}))
    buckets = classify(docs)

    total = len(docs)
    live = sum(1 for d in docs if _is_live_search(d))
    print(f"\nknowledge base: {total} concepts  ({live} registered by live search)")

    _print_bucket("DELETE — fragment ids", buckets["fragment"],
                  "id contains stopwords / prose filler / motion verbs, so it is a "
                  "sentence fragment sliced from a prompt, not a real concept")

    if buckets["manual"]:
        print(f"\n{'=' * 78}")
        print(f"DELETE — human-reviewed wrong mappings  ({len(buckets['manual'])} documents)")
        print("  reason: structurally valid ids whose asset mapping is simply wrong")
        print("=" * 78)
        for doc in sorted(buckets["manual"], key=lambda d: str(d.get("_id", ""))):
            cid = str(doc["_id"])
            print(f"  {cid!r:32} {_MANUAL_JUNK_IDS.get(cid, '')}")
    _print_bucket("DELETE (opt-in) — primitive-only concepts", buckets["primitive_only"],
                  "astronomical/atomic/molecular concepts the architect must build "
                  "from primitives; a mesh here forces the wrong render path")
    _print_bucket("REVIEW — broken asset links", buckets["broken_link"],
                  "asset_src points at a file that is not on disk")

    if buckets["generic"]:
        print(f"\n{'=' * 78}")
        print(f"DELETE (opt-in) — category names / ingester filler  ({len(buckets['generic'])} documents)")
        print("  reason: SEED concepts that match almost any prompt but point at one")
        print("          arbitrary category member; each is redundant with a better singular")
        print("=" * 78)
        for doc in sorted(buckets["generic"], key=lambda d: str(d.get("_id", ""))):
            cid = str(doc["_id"])
            print(f"  {cid!r:16} {_GENERIC_JUNK_IDS.get(cid, '')}")
            print(f"  {'':16} currently -> {doc.get('asset_src')}")

    to_delete = list(buckets["fragment"]) + list(buckets["manual"])
    if args.include_generic:
        to_delete += buckets["generic"]
    if args.include_primitive_only:
        to_delete += buckets["primitive_only"]

    print(f"\n{'-' * 78}")
    print(f"would delete : {len(to_delete)}")
    print(f"would keep   : {total - len(to_delete)}")
    print(f"broken links : {len(buckets['broken_link'])}  (reported only, not deleted)")
    print("-" * 78)

    if not args.apply:
        print("\nDRY RUN — nothing changed. Re-run with --apply to delete.")
        if not args.include_generic and buckets["generic"]:
            print("Add --include-generic to also remove the category-name bucket.")
        if not args.include_primitive_only and buckets["primitive_only"]:
            print("Add --include-primitive-only to also remove that second bucket.")
        return

    if not to_delete:
        print("\nNothing to delete.")
        return

    path = backup(to_delete)
    print(f"\nBackup written: {path}")

    ids = [d["_id"] for d in to_delete]
    result = coll.delete_many({"_id": {"$in": ids}})
    print(f"Deleted {result.deleted_count} documents.")
    print(f"Restore with:  python -m pipeline.kb_cleanup --restore \"{path}\"")


if __name__ == "__main__":
    main()
