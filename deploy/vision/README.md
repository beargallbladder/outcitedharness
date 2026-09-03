# DGX3 datasheet vision

DGX3 (`spark-69c8`) is the page-localization and rendered-page extraction node.
DGX1 remains the protected embedding/search node and is not part of this service.

## Runtime contract

- Model: `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`
- Hugging Face revision:
  `d9748a51ae66354c4dad665aab2c71f26cf2c8cd`
- Served identity: `qwen3-vl-30b-a3b-instruct-fp8`
- Endpoint: `http://spark-69c8:8902/v1`
- Runtime image:
  `sha256:46591c6e4a018d8d197fa246b1e3d682c907654aab4e9402302abb3e6a7dd916`
- Initial runtime mode: eager, one image per request, 32K context, four
  concurrent sequences.

The former `qwen3-coder-next-sglang` container is stopped, not deleted. The
activation command restores it automatically if vision startup fails.

An 8B bootstrap profile can coexist with the coder while the production
checkpoint is staged:

```bash
DGX3_VISION_PROFILE=bootstrap \
  ~/datasheet-vision/scripts/dgx3_datasheet_vision.sh start
```

It is a pipeline smoke/fallback only, not the qualified production extractor.

## Extraction contract

1. Verify the PDF signature and immutable artifact hash.
2. Locate an exact pin-definition table and exact package column.
3. Withhold on ambiguity; never rank a likely page or substitute a related
   package.
4. Render selected pages to pixels.
5. Extract only visibly grounded rows from each page.
6. Merge page-local rows and apply pin-count, duplicate, and grounding checks.
7. Escalate disagreements for frontier comparison.
8. Admit only independently verified corrections as training labels.

Ballout figures, package drawings, alternate-function tables, and cousin
package columns cannot write pin identity.

## Activation and rollback

On DGX3:

```bash
~/datasheet-vision/scripts/dgx3_datasheet_vision.sh preflight
~/datasheet-vision/scripts/dgx3_datasheet_vision.sh activate
~/datasheet-vision/scripts/dgx3_datasheet_vision.sh status
```

Restore the prior coder:

```bash
~/datasheet-vision/scripts/dgx3_datasheet_vision.sh rollback-coder
```

## Modality qualification

The frozen fixture contains only approved datasheets and matching ground-truth
snapshots. The same local model receives the same located pages in two modes:
PyMuPDF text and rendered PNG. Results bind the fixture, evaluator, locator,
model identity, and runtime settings with SHA-256.

```bash
python3 scripts/evaluate_datasheet_modalities.py \
  --fixture fixtures/modality-v1/manifest.json \
  --endpoint http://127.0.0.1:8902/v1 \
  --model qwen3-vl-30b-a3b-instruct-fp8 \
  --model-manifest manifests/qwen3-vl-30b-a3b-instruct-fp8.sha256.json \
  --runtime-image-id sha256:46591c6e4a018d8d197fa246b1e3d682c907654aab4e9402302abb3e6a7dd916 \
  --output evaluations/modality-v1.json
```

CategoryRank and Tapes sources remain excluded.

## Frontier distillation

Frontier comparison requires a completed local `table,image_rows` evaluation.
Every candidate page is independently sent to the configured frontier vision
model. A page becomes training-eligible only when local vision, deterministic
table extraction, and frontier vision agree on every normalized physical row.
The ledger records the image, predictions, proof, model identities, source
revision, and cost before admission.

Admission additionally requires one or more frozen fixtures. Any candidate
whose PDF digest appears in a frozen fixture is rejected before the first API
call or ledger write.

```bash
python3 scripts/compare_datasheet_frontier.py \
  --fixture fixtures/train-candidates-v1/manifest.json \
  --local-evaluation evaluations/train-candidates-local.json \
  --output evaluations/train-candidates-frontier.json \
  --split-role candidate \
  --frozen-fixture fixtures/modality-v1/manifest.json \
  --capture-ledger --admit-consensus

python3 scripts/build_datasheet_frontier_dataset.py \
  --destination datasets/datasheet-frontier-v1 \
  --frozen-fixture fixtures/modality-v1/manifest.json
```

The five-example Qwen3-VL 8B plumbing smoke uses
`deploy/training/llamafactory_datasheet_frontier_vision_smoke.yaml`. It is not
promotion evidence; promotion still requires a sealed base-versus-adapter run
on the untouched frozen fixture.
