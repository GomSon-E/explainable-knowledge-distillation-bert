# Project Agent Rules

## Project goal

Implement and evaluate explainable knowledge distillation for **BERT-based English question classification**.

- Primary dataset candidate: TREC-6 (six question classes).
- Use TREC-6 inputs up to BERT's maximum **512 tokens**; exclude longer examples.
- Teacher: fine-tuned 12-layer BERT.
- Students: 10-layer, 8-layer, and 6-layer BERT.
- Compare baseline, standard KD, KD + IG explanation loss, and KD + LRP explanation loss as defined in `03_experiment_plan.md`.

## Source-of-truth documents

Read these before making design or implementation decisions:

1. `01_video_notes.md` — mentor direction and terminology
2. `02_assignment_brief.md` — assignment requirements and project definition
3. `03_experiment_plan.md` — experimental conditions and required outputs
4. `04_presentation_plan.md` — presentation narrative and required visuals

If a requested implementation conflicts with these files, point out the conflict and ask before changing the experimental design.

## Development versus execution environment

- Code is authored and edited locally through VS Code/Codex.
- **All real training, GPU evaluation, XAI computation, and result generation must run in Google Colab with a GPU runtime.**
- Do not assume this local machine has CUDA, a GPU, enough RAM, or persistent experiment storage.
- Keep code portable: it must work after the project folder is uploaded or cloned into Colab.
- Prefer normal Python modules/scripts plus a thin Colab notebook runner. Avoid logic that only works inside VS Code.

## Colab compatibility requirements

- At the top of the notebook or entry script, check and report the active PyTorch device (`cuda` if available).
- Put installation commands in a clearly separated Colab setup cell or `requirements.txt`; pin versions when an incompatibility is discovered.
- Make data, checkpoints, metrics, plots, and token-attribution outputs save to configurable paths.
- Default Colab paths may use `/content`; support an optional Google Drive root for persistent artifacts.
- Never hard-code an absolute local VS Code path.
- Support resuming from existing Teacher/Student checkpoints because Colab sessions can disconnect.
- Add lightweight smoke-test options: tiny data subset, few steps/epochs, and one batch attribution calculation.

## Implementation and experiment rules

- Use one fixed 12-layer Teacher for every KD condition; do not perform chained Teacher-to-Student distillation.
- Freeze the Teacher while training Students.
- Use identical data splits, preprocessing, tokenizer, seed policy, evaluation set, and timing protocol across comparable models.
- Keep the 13 experimental conditions separate and name checkpoints/metrics unambiguously by depth and method.
- Compute classification metrics: Accuracy, Precision, Recall, and Macro F1.
- Record parameter count, model file size, and classification-only inference time. Do not include IG/LRP explanation time in normal inference latency.
- Log all applicable loss components separately: task loss, KD loss, IG explanation loss, and LRP explanation loss.
- Explainability results must be reproducible: save per-token attribution scores with token strings and the predicted/target class.

## XAI-specific rules

- IG and LRP are not visualization-only features. For their respective KD conditions, attribution mismatch must be included in the Student training loss.
- First confirm on a tiny batch that the explanation loss has a differentiable path to Student parameters and produces nonzero gradients.
- Compute Teacher and Student attributions for the same target class and align tokens before comparing them.
- Exclude or explicitly handle special tokens and padding when computing attribution similarity/loss.
- Generate separate results for IG KD and LRP KD; never label one method's results as the other.
- Save visualization-ready token-importance examples for Teacher, baseline, standard KD, IG KD, and LRP KD.

## Deliverable discipline

- Do not claim an experiment was run unless its saved metrics and artifacts exist.
- Keep a machine-readable experiment summary (CSV or JSON) containing configuration, seed, metrics, efficiency measurements, and artifact paths.
- Create plots and tables that directly support the planned presentation: performance, efficiency, attribution similarity, and qualitative token-importance examples.
- When a computation is too costly for Colab, report the limitation and provide the smallest valid fallback; do not silently skip it.
