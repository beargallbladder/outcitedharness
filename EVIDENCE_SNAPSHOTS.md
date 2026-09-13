# Evidence Snapshot Provenance — Design Plan

Status: planning, 2026-09-12. Owner: m5-opencode. CR conference: proposed
(mail `evidence-snapshot-provenance-proposal-20260912`).

## 1. The problem this solves

1. **Trust**: claims are currently verified by re-derivation from the PDF
   (vault access required) or by trusting the extractor. A snapshot of the
   exact printed evidence makes every claim independently checkable by
   anyone holding the image — no PDF, no URL, no harness.
2. **Moat protection**: provenance-as-URL hands a partner our entire
   sourcing map (vendor URL grammars, fetch lanes). Provenance-as-image
   leaks nothing reusable. The pdf_sha256 stays in the vault.
3. **Link rot**: URLs 404 (infineon /dgdl/, renesas, this week alone) and
   volumes unmount (MACMOBILE, twice this session). Frozen renders survive
   both.
4. **Partial ground truthing**: claim + snapshot = a reviewable unit. A
   human or second model adjudicates "the image says pin 1 = VBAT" without
   trusting us. Snapshots become GT anchors for the reconciliation queues.

## 2. The artifact: the evidence bundle

Every extracted row claim carries:

```json
{
  "claim_id": "pin-<doc_sha16>-p<page>-<pkg>-<pin_no>",
  "kind": "pin_identity",
  "package": "LQFP48",
  "pin_no": "7",
  "name": "VBAT",
  "extraction_lane": "word_columns | vision",
  "extraction_method": "pymupdf-word-columns-v3 | glm-5v-turbo+grounded",
  "evidence": {
    "document_sha256": "<vault key; shared as opaque doc-id at partner tier>",
    "page_1based": 19,
    "page_render_sha256": "<sha256 of the frozen page PNG>",
    "render": {"dpi": 144, "renderer": "pymupdf", "renderer_version": "<pinned>"},
    "row_band_bbox": [x0, y0, x1, y1],
    "crop_sha256": "<sha256 of the materialized row-band crop>"
  }
}
```

- **Page render**: full-page PNG at 144 DPI, stored once per
  (doc, page), content-addressed by its sha256. ~200 KB/page; 20,973 pin
  pages ≈ 4.2 GB on the 4 TB vault.
- **Row-band crop**: the horizontal band spanning the row's identifier and
  name cells, cropped from the frozen render. Materialized lazily from
  (render + bbox) at export time; the crop sha256 recorded in the claim
  makes the derivation tamper-evident (anyone with the PDF can re-render,
  re-crop, and byte-compare).
- **Deterministic lanes** (word_columns): bboxes already exist per claim —
  crops derive directly.
- **Vision lane** (GLM): the model returns no coordinates. The grounding
  pass (which we already run) locates each extracted pin name/number in the
  page's word list — that match supplies the bbox. Vision pins that fail
  grounding get page-level evidence only, explicitly marked
  `evidence_granularity: "page"` — never silently upgraded.

## 3. Verification tiers

| Tier | Holder sees | Check |
|---|---|---|
| Vault (us) | everything | full re-derivation |
| Partner (CR) | claims + crops + doc-sha, **no URLs** | compare claim against crop image; optionally re-render if they later acquire the PDF by sha |
| Public | claims + doc-sha only | trust anchor, no crop distribution (copyright footprint of expression) |

Tamper-evidence: claim's crop_sha256 + row_band_bbox + page_render_sha256;
re-derivation with the pinned renderer version must byte-match. Renderer
version is recorded in every manifest; re-renders under a different version
are compared by content, flagged on mismatch.

## 4. Retrofit paths (the ~180K rows already extracted this week)

- **Phase R1 — page-render vault** (post-scrub): re-render all 20,973 pin
  pages deterministically, content-address, manifest with render params.
  ~35 min at 8 workers, 4.2 GB. Survives volume unmounts.
- **Phase R2 — deterministic-claim augmentation**: backfill script over
  sweep v3 extractions: crop row bands from vault renders, compute crop
  shas, emit evidence bundles for all 54,668 word_columns rows.
- **Phase R3 — vision-claim location**: extend the grounding pass to record
  the matched word bboxes per pin; emit bundles for the ~127K scrub rows.
  Ungrounded pins stay page-granularity.
- **Phase R4 — GT reconciliation v4 as evidence bundles**: the 258
  GT-extension rows and the genuine-discrepancy queue become
  human-reviewable bundles: crop + extracted row + GT row side by side.
  This is the "partial ground truthing" product.
- **Phase R5 — tiered export builder**: drop materializer that emits
  vault/partner/public shapes; partner drops carry crop files + SHA256SUMS.

## 5. Going forward

- `run_datasheet_structural_extraction.py` already seals focused images —
  align its evidence contract with the bundle schema (same fields, same
  content addressing).
- New extraction lanes emit bundles natively (the scrub gets patched to
  save its in-memory renders instead of discarding them — one-line change
  at next restart).
- Storage growth: future weekly waves at current volume ≈ 1-2 GB/week of
  renders — negligible on 4 TB; manifest discipline is the real cost.

## 6. Risks and open questions

1. **Render determinism** is version-bound (pymupdf upgrades may shift
   anti-aliasing). Mitigation: pin renderer version per manifest; treat
   cross-version mismatch as a re-verification event, not silent trust.
2. **Crop integrity = trust in the cropper** — mitigated by the re-crop
   byte-compare path and by the deterministic lane being the primary
   extractor (the vision lane's crops carry the grounding-match provenance).
3. **Copyright**: crops are closer to vendor expression than normalized
   facts; partner tier only, public tier stays claims + doc-sha. (CR
   conference will confirm their ingest wants crops at all.)
4. **Completeness is out of scope for snapshots** — a crop proves the row
   it shows, not the absence of missed rows. Set-level witnesses (bogey,
   invariants, cross-lane agreement) remain the completeness instruments.
5. **Reverse image search** can map a crop back to its source document.
   Accepted at partner tier; noted for the public tier decision.

## 7. Sequencing

Phase 0 (now): this plan + CR proposal mail.
Phase R1 fires when the vision scrub's tail completes (render workers reuse
the freed machine). R2-R3 same day. R4 with the reconciliation re-run
(already pending the pairing fix). R5 before the next partner drop.
