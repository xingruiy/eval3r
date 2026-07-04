"""Canonical protocol hashing.

Process (per ``.agent/schema.md`` "Canonical hashing" and ``.agent/reproducibility.md``
"Protocol hashing"):

    validate into EvalProtocol -> serialize to JSON-able dict -> strip non-semantic
    content -> serialize to canonical JSON (sorted keys, no whitespace) -> UTF-8 ->
    sha256 -> prefix with ``sha256:``.

The hash must change when evaluation behavior changes and must NOT change because of
YAML comments, key ordering, whitespace, human notes, or report-only preferences.

The set of non-semantic fields excluded from the hash is **explicit** here and is
covered by tests, as required by ``.agent/reproducibility.md``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from eval3r.core.protocol import EvalProtocol

# Keys removed at ANY nesting depth: free-form human prose that documents intent
# but does not change what is computed.
NON_SEMANTIC_RECURSIVE_KEYS: frozenset[str] = frozenset({"notes", "reason"})

# Top-level protocol keys removed: report-only output preferences that select which
# artifacts/formats are written but never change a metric value.
NON_SEMANTIC_TOPLEVEL_KEYS: frozenset[str] = frozenset({"reporting"})


def _strip_recursive(value: Any) -> Any:
    """Return a copy of ``value`` with all NON_SEMANTIC_RECURSIVE_KEYS removed."""
    if isinstance(value, dict):
        return {
            k: _strip_recursive(v)
            for k, v in value.items()
            if k not in NON_SEMANTIC_RECURSIVE_KEYS
        }
    if isinstance(value, list):
        return [_strip_recursive(v) for v in value]
    return value


def canonical_protocol_payload(protocol: EvalProtocol) -> dict[str, Any]:
    """The semantic, JSON-able dict that is hashed for ``protocol``.

    Exposed (and tested) so the exclusion policy is inspectable, not hidden inside
    the hash function.
    """
    payload = protocol.model_dump(mode="json")
    for key in NON_SEMANTIC_TOPLEVEL_KEYS:
        payload.pop(key, None)
    stripped = _strip_recursive(payload)
    assert isinstance(stripped, dict)
    return stripped


def canonical_json(payload: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, non-ASCII preserved."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def protocol_hash(protocol: EvalProtocol) -> str:
    """Return the canonical ``sha256:...`` hash of ``protocol``'s semantic content."""
    payload = canonical_protocol_payload(protocol)
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
