# Confidence policies

Predictions may come with confidence values (per point) or confidence maps (per pixel).
How they are used is a **protocol decision**, never a method-specific hidden threshold:
a method scored with its own tuned cutoff is not comparable to one scored raw.

## Policies

```yaml
confidence:
  policy: none | method_default | threshold | percentile | top_k_fraction
  threshold: 0.5          # for policy: threshold
  percentile: 90          # for policy: percentile
  top_k_fraction: 0.9     # for policy: top_k_fraction
  allow_override: false
```

- `none` — confidence is ignored; all points/pixels are scored.
- `method_default` — the method's declared native threshold (from the manifest) is
  applied; the value used is recorded.
- `threshold` — keep predictions with confidence above a protocol-pinned absolute value.
- `percentile` — keep predictions above a protocol-pinned confidence percentile.
- `top_k_fraction` — keep the most confident fraction of predictions.

The policy and its parameter are part of the protocol hash, and the effective filtering
is recorded in results (`valid_fraction` shows what survived).

## Self-filtered predictions

If a method filters its own output by confidence **before export**, its manifest must
declare that (`confidence.self_filtered: true`, with `native_threshold` when known).
Reports distinguish uniform protocol filtering from method-native filtering — the two
produce numbers with different meanings even under the same protocol policy.

A differing confidence policy between two runs is one of the comparability warnings
`e3r diff` raises.
