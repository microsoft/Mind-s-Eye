# Minds_Eye

Minds_Eye is a synthetic visual reasoning benchmark generator and evaluation toolkit.  
It creates image-based multiple-choice puzzles across several cognitive/perceptual skills and includes a model-evaluation pipeline for vision-language models (VLMs).

The repository is organized around two workflows:

1. **Dataset generation** (create puzzle images + JSON annotations)
2. **Model evaluation** (run VLMs on the generated tasks and score responses)

---

## Table of Contents

- [What this repo contains](#what-this-repo-contains)
- [Task families](#task-families)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Generate datasets](#generate-datasets)
- [Run evaluation](#run-evaluation)
- [Annotation formats](#annotation-formats)
- [Important implementation notes](#important-implementation-notes)
- [Extending the benchmark](#extending-the-benchmark)
- [Troubleshooting](#troubleshooting)

---

## What this repo contains

Minds_Eye provides:

- Procedural generators for **8 visual reasoning task families**.
- Pre-generated datasets under `data/`.
- Annotation JSON files with question text and answer keys.
- A VLM evaluation pipeline with support for multiple Hugging Face / InternVL models.
- A lightweight analysis script for aggregating judged accuracy.

---

## Task families

The project currently includes these task categories:

1. **Slippage** (`data/slippage`)  
   Concept violation detection (spacing, alignment, number, enclosure, symmetry, etc.).

2. **Abstract Analogical Reasoning** (`data/abstract`)  
   Identify the odd item among abstract relation-based patterns.

3. **Mental Rotation** (`data/mental_rotation`)  
   Select the option that is a valid rotation of an original 3D block-like object.

4. **Mental Composition** (`data/mental_composition`)  
   Match a 2D net to its corresponding 3D assembled shape.

5. **Paper Folding** (`data/paper_folding`)  
   Predict unfolded hole patterns from folding + punch sequences.

6. **Dynamic Isomorphism** (`data/dynamic_isomorph`)  
   Infer temporal transformation continuation and select the next state.

7. **Symmetric Structures** (`data/symmetric_isomorph`)  
   Find the structure that breaks symmetry constraints.

8. **Hierarchical Isomorphism** (`data/hierarchial_isomorph`)  
   Detect the option violating recursive/hierarchical structural regularities.

---

## Repository structure

```text
Minds_Eye/
├── data/                          # Pre-generated datasets (images + annotations)
│   ├── abstract/
│   ├── dynamic_isomorph/
│   ├── hierarchial_isomorph/
│   ├── mental_composition/
│   ├── mental_rotation/
│   ├── paper_folding/
│   ├── slippage/
│   └── symmetric_isomorph/
├── lib/                           # Core generation logic per task family
│   ├── abstract_reasoning.py
│   ├── dynamic_isomorphism.py
│   ├── hierarchial.py
│   ├── mental_composition_v2.py
│   ├── mental_rotation_v2.py
│   ├── paper_folding_v2.py
│   ├── slippage.py
│   └── symmetric_structres.py
├── evaluation/
│   ├── evaluation_pipeline.py     # End-to-end model evaluation
│   ├── analysis.py                # Aggregate judged scores
│   └── internvl35.py              # InternVL 3.5 loader/inference helper
└── generate_dataset.py            # Main dataset generation entrypoint
```

---

## Setup

> Recommended: Python 3.10+ in a fresh virtual environment.

### 1) Create and activate environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

Depending on your GPU stack and model choices, you may also need:

- CUDA-enabled PyTorch build
- `accelerate`
- vendor-specific libraries for large multimodal checkpoints

---

## Generate datasets

`generate_dataset.py` is the main orchestrator and can generate all task families.

### Quick run

Update the output path in `generate_dataset.py` (`Dataset_dir`) or call individual generation functions from a custom script.

Example minimal driver:

```python
from generate_dataset import (
    generate_slippage,
    generate_abstract_reasoning,
    generate_mrt,
    generate_mental_composition,
    generate_paper_folding,
    generate_dynamic_isomorphism,
    generate_symmetric_structures,
    generate_hierarchial_structures,
)

OUT = "./data"
N = 100

generate_slippage(OUT, N)
generate_abstract_reasoning(OUT, N)
generate_mrt(OUT, N)
generate_mental_composition(OUT, N)
generate_paper_folding(OUT, N)
generate_dynamic_isomorphism(OUT, N)
generate_symmetric_structures(OUT, N)
generate_hierarchial_structures(OUT, N)
```

For abstract analogical generation, set these environment variables first:

```bash
export BONGARD_SHAPES_PATH=/path/to/human_designed_shapes.tsv
export BONGARD_ATTRIBUTES_PATH=/path/to/human_designed_shapes_attributes.tsv
```

### Generation outputs

Each task directory typically contains:

- `*.png` puzzle images
- `annotations.json` metadata with question + target label(s)

---

## Run evaluation

`evaluation/evaluation_pipeline.py` evaluates models over dataset JSONs and writes CSV/JSON outputs.

### Supported model keys

Defined in `MODELS`:

- `llama`
- `internvl3_5`
- `qwen7b`
- `qwen3b`
- `llava`
- `Idefics3`

### Basic usage

1. Configure environment variables (optional defaults shown):
   - `MINDS_EYE_DATASET_ROOT` (default: `./data`)
   - `MINDS_EYE_EVAL_OUTPUT_DIR` (default: `./model_evaluation`)
   - `MINDS_EYE_DEVICE` (default: `cuda:0`)
   - `HF_TOKEN` (required for gated Hugging Face models)
   - `MINDS_EYE_OPTION_EXTRACTOR_MODEL` (default: `gemma3:4b`)
   - `MINDS_EYE_OLLAMA_BASE_URL` (default: `http://localhost:11434`)
2. Ensure model checkpoints are available.
3. Run:

```bash
python evaluation/evaluation_pipeline.py
```

### Outputs

For each model and step:

- Per-task CSV files: `*_judged_<task>.csv`
- Aggregate CSV: `*_phi3_judged_all_tasks.csv`
- Aggregate JSON: `*_juded_model_answer.json`

### Score extraction

The pipeline uses a custom `OptionExtractionMetric` (via DeepEval + Ollama model) to parse a final option letter and compare against ground truth.

---

## Annotation formats

All tasks store metadata as JSON dicts keyed by image filename.

Common fields include:

- `question`: natural-language instruction
- task-specific answer key field, e.g.:
  - `answer`
  - `correct_option`
  - `fifth_label`
  - `violation`
  - `asymmetric_label`

The evaluation mapping in `ANSWER_KEY` determines which field is used as ground truth per dataset.

---

## Extending the benchmark

To add a new task family:

1. Add generation logic in `lib/<new_task>.py`.
2. Expose a wrapper in `generate_dataset.py`.
3. Save outputs in `data/<new_task>/` with `annotations.json`.
4. Register in evaluation:
   - add dataset JSON path in `JSON_LIST`
   - add answer-key mapping in `ANSWER_KEY`
   - ensure model prompt format supports the task

---

## Troubleshooting

### CUDA / OOM errors

- Reduce model size.
- Use fewer concurrent models.
- Lower generation length (`max_new_tokens`).

### `internvl3_5` loading issues

- Confirm trust-remote-code compatibility.
- Verify GPU count and memory assumptions in `internvl35.py`.

### Empty or partial evaluation outputs

- Confirm dataset JSON paths are valid.
- Check that image filenames in annotations match actual files.
- Remove debug `break` for full-dataset traversal.
