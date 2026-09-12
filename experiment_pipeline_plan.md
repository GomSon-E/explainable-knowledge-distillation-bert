# Explainable BERT KD Experiment Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Google Colab GPU에서 TREC-6 기반 Teacher 1개와 Student 12개를 동일한 데이터·재현성·평가 계약으로 학습하고, 성능·효율·설명 정합성 결과를 재현 가능하게 저장한다.

**Architecture:** 학습 로직은 `src/explainable_kd`의 일반 Python 모듈에 두고 Colab notebook은 설치·장치 확인·CLI 호출만 담당한다. `configs/experiments.yaml`의 불변 `experiment_id`와 seed를 run key로 사용하며, 모든 단계는 파일 기반 artifact를 통해 연결되어 Colab 세션 중단 후 재개할 수 있다.

**Tech Stack:** Python 3, PyTorch, Hugging Face Transformers/Datasets, Captum 또는 검증된 BERT LRP 구현, scikit-learn, pandas, matplotlib/seaborn, PyYAML, pytest

**Spec:** `AGENTS.md`, `01_video_notes.md`, `02_assignment_brief.md`, `03_experiment_plan.md`, `04_presentation_plan.md`, `README.md`

## Global Constraints

- 실제 학습, GPU 평가, XAI 계산, 결과 생성은 Google Colab GPU에서만 수행한다.
- 데이터셋은 TREC-6, tokenizer와 pretrained model은 `google-bert/bert-base-cased`, tokenized maximum length는 10이다.
- 12층 `teacher_d12_supervised` 하나만 모든 KD 조건에 사용하며 Teacher는 Student 학습 중 freeze한다.
- 10·8·6층 각각 baseline, KD, IG KD, LRP KD를 분리하여 총 13개 조건을 유지한다.
- 모든 비교 조건은 동일한 split manifest, tokenizer, seed policy, evaluation set, timing protocol을 사용한다.
- 분류 지표는 Accuracy, Macro Precision, Macro Recall, Macro F1을 기록한다.
- 효율 지표는 parameter count, model file size, classification-only inference latency를 기록하며 XAI 시간은 제외한다.
- IG/LRP loss는 시각화용 후처리가 아니라 Student parameter까지 미분 가능한 학습 손실이어야 한다.
- Teacher와 Student attribution은 같은 target class와 정렬된 token mask를 사용하며 special token과 padding을 제외한다.
- 실행 완료는 터미널 메시지가 아니라 checkpoint와 machine-readable metrics artifact의 존재로 판정한다.
- 현재 디렉터리는 Git 저장소가 아니다. 아래 커밋 단계는 저장소가 초기화되거나 상위 저장소에 포함된 뒤 수행한다.

---

## 기능 플로우

```text
base config + experiment registry
  -> deterministic TREC-6 split/tokenization -> data fingerprint
  -> teacher_d12_supervised best checkpoint
  -> baseline 3개 (Teacher 미사용)
  -> standard KD 3개 (동일 Teacher, frozen)
  -> IG gradient gate -> IG KD 3개
  -> LRP gradient gate -> LRP KD 3개
  -> 13개 best checkpoint 공통 평가
  -> run metrics + token attribution JSONL
  -> summary CSV/JSON + 발표용 표/그래프/사례
```

## 변경 대상 파일 목록

| Step | 기능 | 주요 파일 | 자동 테스트 |
|---:|---|---|:---:|
| 1 | 패키징·설정 계약 | `pyproject.toml`, `common/config.py`, `configs/*.yaml` | ✓ |
| 2 | 데이터 처리 | `data/pipeline.py`, `tests/test_data_pipeline.py` | ✓ |
| 3 | 공통 seed·metrics·checkpoint | `common/{seed,metrics,checkpoint}.py` | ✓ |
| 4 | 모델 구성과 Teacher/Baseline | `training/{teacher,baseline}.py` | ✓ |
| 5 | 일반 KD | `training/kd.py` | ✓ |
| 6 | IG KD | `xai/ig.py`, `training/kd.py` | ✓ |
| 7 | LRP KD | `xai/lrp.py`, `training/kd.py` | ✓ |
| 8 | CLI·Colab·재개 | `cli.py`, `notebooks/run_experiments_colab.ipynb` | ✓ |
| 9 | 전체 평가·시각화 | `evaluation/{evaluate,visualize}.py` | ✓ |
| 10 | 13개 조건 smoke 및 본 실행 | 설정과 생성 artifact | ✓ |

---

### Step 1 `[Claude]` `[✓]`: 패키징과 설정 계약 구현

> **구현 포인트:** 실험 ID registry를 코드에서 중복 정의하지 않는다. 설정 로더가 13개 ID의 유일성, 허용 depth/method 조합, 모든 KD 조건의 동일 Teacher 참조를 시작 전에 검증해야 한다.

**Files:**

- Create: `pyproject.toml`
- Modify: `src/explainable_kd/common/config.py`
- Modify: `configs/base.yaml`
- Modify: `configs/smoke.yaml`
- Modify: `configs/experiments.yaml`
- Create: `tests/test_config.py`

**Interfaces:**

- Consumes: YAML 파일과 CLI override `artifact_root`, `seed`, `resume`, `smoke_test`
- Produces: `ExperimentConfig`, `ExperimentSpec`, `load_config(base_path, overlay_path=None, overrides=None)`, `get_experiment(experiment_id)`

- [ ] `ExperimentSpec`에 `experiment_id: str`, `role: Literal["teacher", "student"]`, `depth: Literal[6, 8, 10, 12]`, `method: Literal["supervised", "baseline", "kd", "ig_kd", "lrp_kd"]`, `teacher_ref: str | None`를 정의한다.
- [ ] 13개 canonical ID가 정확히 한 번씩 존재하고 Teacher가 하나인지 검사하는 실패 테스트를 작성한다.
- [ ] KD·IG KD·LRP KD의 `teacher_ref`가 모두 `teacher_d12_supervised`인지, baseline은 `teacher_ref=null`인지 검사한다.
- [ ] 상대 artifact root는 project root 기준, Colab Drive override는 전달된 절대 경로 기준으로 resolve하되 로컬 경로를 기본값으로 하드코딩하지 않는다.
- [ ] `pytest tests/test_config.py -v`를 실행하여 잘못된 ID, 중복 ID, 잘못된 Teacher 참조가 명확한 `ValueError`로 실패하는지 확인한다.

**검증 방법:** `python -m explainable_kd.cli list-experiments`가 13개 ID를 registry 순서대로 출력하고 exit code 0을 반환해야 한다.

**커밋 메시지:** `FEAT: Add validated experiment configuration registry`

**설계 근거:** YAML registry 하나를 source of truth로 사용하면 notebook, checkpoint path, summary row에서 명칭이 갈라지는 것을 방지할 수 있다. Hydra 같은 추가 설정 프레임워크는 현재 규모에 필요하지 않으므로 사용하지 않는다.

---

### Step 2 `[Claude]` `[✓]`: 결정적 TREC-6 데이터 처리 구현

> **구현 포인트:** 모든 조건이 같은 질문과 label mapping을 보도록 split 결과 자체를 manifest로 고정한다. `max_length=10`은 tokenizer 결과에 적용하며, filter/truncate 정책과 보존 표본 수를 manifest에 기록한다.

**Files:**

- Modify: `src/explainable_kd/data/pipeline.py`
- Create: `tests/test_data_pipeline.py`

**Interfaces:**

- Consumes: `ExperimentConfig.data`, seed, Hugging Face TREC dataset
- Produces: `prepare_dataset(config) -> DatasetDict`, `DataManifest`, `data_fingerprint: str`

- [ ] TREC coarse label을 6개 고정 순서 `ABBR, ENTY, DESC, HUM, LOC, NUM`으로 매핑하는 테스트를 작성한다.
- [ ] 동일 seed/config가 같은 example IDs, split counts, fingerprint를 만드는 테스트를 작성한다.
- [ ] tokenize 후 attention mask 길이가 10을 넘지 않고 special token 처리 정책이 manifest에 남는 테스트를 작성한다.
- [ ] raw dataset revision, tokenizer name/revision, label map, length policy, split example IDs, counts를 `artifacts/data/split_manifest.json`에 저장한다.
- [ ] 처리 데이터는 `artifacts/data/processed/<data_fingerprint>/`에 저장하고 기존 fingerprint가 일치하면 재사용한다.
- [ ] smoke overlay가 train 32, validation 16, test 16개 이하를 선택하되 원본 split 순서를 훼손하지 않는지 확인한다.

**검증 방법:** Colab CPU에서도 `prepare-data --smoke`를 실행해 manifest와 processed dataset이 생성되고, 두 번째 실행이 같은 fingerprint를 재사용해야 한다.

**커밋 메시지:** `FEAT: Add deterministic TREC-6 preprocessing`

**설계 근거:** 저장된 example ID manifest가 있어야 13개 조건의 비교 가능성을 사후에 검증할 수 있다. 단순 seed 기록만으로는 dataset revision이나 filtering 변화까지 탐지할 수 없다.

---

### Step 3 `[Claude]` `[✓]`: 공통 seed·metrics·checkpoint 유틸 구현

> **구현 포인트:** checkpoint는 `(experiment_id, seed)` 밖을 검색하지 않으며 `last`는 재개용, `best`는 최종 평가용으로 역할을 분리한다. 적용되지 않는 loss는 `null`로 직렬화한다.

**Files:**

- Modify: `src/explainable_kd/common/seed.py`
- Modify: `src/explainable_kd/common/metrics.py`
- Modify: `src/explainable_kd/common/checkpoint.py`
- Create: `tests/test_seed.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_checkpoint.py`

**Interfaces:**

- Produces: `seed_everything(seed, deterministic)`, `seed_worker(worker_id)`, `classification_metrics(y_true, y_pred)`, `count_parameters(model)`, `measure_model_size(path)`, `CheckpointManager(artifact_root, experiment_id, seed)`
- Checkpoint payload: model state, optimizer state, scheduler state, epoch/global step, best metric, resolved config, config hash, data fingerprint, RNG states

- [ ] 동일 seed에서 Python/NumPy/PyTorch 난수열과 DataLoader 순서가 재현되는 테스트를 작성한다.
- [ ] 알려진 label/prediction fixture로 Accuracy와 macro Precision/Recall/F1을 검산한다.
- [ ] checkpoint 경로가 `checkpoints/<experiment_id>/seed_<seed>/{best,last}`인지 검사한다.
- [ ] 저장 후 RNG와 optimizer state를 복원해 다음 step이 연속 실행과 같은 값을 내는 테스트를 작성한다.
- [ ] config hash 또는 data fingerprint가 다른 checkpoint 재개를 기본적으로 거부하고, 명시적 override 없이는 진행하지 않게 한다.
- [ ] train history JSONL schema에서 각 loss 필드와 run metadata가 유지되는지 검사한다.

**검증 방법:** `pytest tests/test_seed.py tests/test_metrics.py tests/test_checkpoint.py -v`가 CPU 환경에서 통과해야 한다.

**커밋 메시지:** `FEAT: Add reproducible experiment utilities`

**설계 근거:** 공통 유틸을 먼저 고정해야 이후 모든 trainer가 같은 저장·재개·지표 규칙을 강제로 사용한다. 각 trainer별 저장 코드는 경로 충돌과 resume 편차를 만든다.

---

### Step 4 `[Claude]` `[✓]`: 모델 구성과 Teacher·Baseline 학습 구현

> **구현 포인트:** Student는 같은 BERT initialization 정책에서 encoder layer 수만 10/8/6으로 변경한다. baseline trainer는 Teacher 경로를 인자로 받지 않아 구조적으로 Teacher 사용이 불가능해야 한다.

**Files:**

- Create: `src/explainable_kd/training/model_factory.py`
- Modify: `src/explainable_kd/training/teacher.py`
- Modify: `src/explainable_kd/training/baseline.py`
- Create: `tests/test_model_factory.py`
- Create: `tests/test_supervised_training.py`

**Interfaces:**

- Produces: `build_classifier(depth, num_labels)`, `train_teacher(config, datasets, checkpoint_manager)`, `train_baseline(spec, config, datasets, checkpoint_manager)`
- Training output: best/last checkpoint and epoch-level JSONL with `task_loss`, validation metrics, run metadata

- [ ] factory가 정확히 12/10/8/6 encoder layers와 6-label head를 만드는 테스트를 작성한다.
- [ ] tiny synthetic batch에서 supervised loss backward가 finite·nonzero Student gradient를 만드는 테스트를 작성한다.
- [ ] Teacher와 baseline이 labels-only cross entropy를 사용하고 KD/XAI loss를 `null`로 기록하는지 검사한다.
- [ ] validation Macro F1 기준 best와 매 step/epoch의 last checkpoint 저장을 연결한다.
- [ ] resume 시 epoch/global step과 optimizer/scheduler/RNG state가 이어지는지 smoke test한다.
- [ ] Colab smoke data로 Teacher 2 steps와 baseline depth별 2 steps를 실행한다.

**검증 방법:** smoke run마다 `best/`, `last/`, `train_history.jsonl`가 존재하고 model config의 layer count가 experiment registry와 일치해야 한다.

**커밋 메시지:** `FEAT: Add Teacher and Student baseline training`

**설계 근거:** 모델 생성 정책을 factory 한 곳에 두어 depth 외 설정 차이로 인한 교란을 막는다. Teacher와 baseline 흐름은 손실 입력이 다르므로 별도 entry function으로 유지한다.

---

### Step 5 `[Claude]` `[✓]`: 일반 Knowledge Distillation 구현

> **구현 포인트:** Teacher는 `eval()`과 `requires_grad_(False)`를 적용하고 모든 3개 depth가 동일 best checkpoint를 읽는다. KD soft target에는 동일 temperature를 적용하고 `T²` scaling 정책을 config와 로그에 기록한다.

**Files:**

- Modify: `src/explainable_kd/training/kd.py`
- Create: `tests/test_kd_training.py`

**Interfaces:**

- Produces: `kd_loss(student_logits, teacher_logits, temperature)`, `train_kd(spec, config, datasets, teacher_checkpoint, checkpoint_manager, explanation_provider=None)`
- Loss contract: `total_loss = alpha * task_loss + beta * kd_loss` for method `kd`

- [ ] 동일 logits에서 KD loss가 최소이고 다른 logits에서 증가하는 테스트를 작성한다.
- [ ] backward 후 Student gradient는 nonzero이고 Teacher parameter의 `.grad`는 모두 `None`인지 검사한다.
- [ ] teacher checkpoint의 experiment ID가 `teacher_d12_supervised`가 아니면 시작 전에 실패시킨다.
- [ ] depth 10/8/6 smoke run이 각각 독립 경로에 checkpoint와 `task_loss`, `kd_loss`, `total_loss`를 기록하도록 연결한다.
- [ ] temperature, alpha, beta와 Teacher checkpoint hash를 resolved config 및 train log에 저장한다.

**검증 방법:** `pytest tests/test_kd_training.py -v`와 3개 depth 2-step Colab smoke run을 수행한다. Teacher checkpoint hash가 세 run에서 동일해야 한다.

**커밋 메시지:** `FEAT: Add standard knowledge distillation training`

**설계 근거:** 설명 없는 KD를 먼저 독립 검증해야 IG/LRP 조건의 변화가 설명 손실에서 왔는지 비교할 수 있다.

---

### Step 6 `[Claude]` `[✓]`: Integrated Gradients KD 구현

> **구현 포인트:** Teacher와 Student는 같은 target class를 설명하며 tokenizer가 동일하므로 같은 input positions와 `valid_mask`를 사용한다. Student attribution 계산에 필요한 higher-order gradient graph를 끊지 않는다.

**Files:**

- Modify: `src/explainable_kd/xai/ig.py`
- Modify: `src/explainable_kd/training/kd.py`
- Create: `tests/test_ig.py`

**Interfaces:**

- Produces: `integrated_gradients(model, batch, target_class, steps, create_graph) -> AttributionBatch`, `ig_explanation_loss(teacher_attr, student_attr, valid_mask)`, `verify_ig_gradient_path(...) -> GradientCheck`
- Loss contract: `total_loss = alpha * task_loss + beta * kd_loss + gamma * ig_loss`

- [ ] baseline embedding과 integration step 수를 명시하고 completeness sanity check를 tiny model에서 작성한다.
- [ ] `[CLS]`, `[SEP]`, padding이 loss에서 제외되고 실제 token position만 normalization되는지 검사한다.
- [ ] Teacher/Student에 서로 다른 target class가 전달되면 실패하는 테스트를 작성한다.
- [ ] one-batch `total_loss.backward()` 후 Student parameter gradient norm이 finite이고 0보다 큰지 검사한다.
- [ ] Teacher attribution은 detach하되 Student attribution graph는 유지하고 Teacher gradient는 생성되지 않음을 검사한다.
- [ ] gradient gate 성공 JSON이 없으면 10/8/6층 IG KD 본 학습 명령이 거부되게 한다.
- [ ] attribution JSONL에 tokens, scores, valid mask, target/predicted class를 저장한다.

**검증 방법:** Colab에서 `verify-xai-gradients --method ig --config configs/smoke.yaml`을 먼저 실행한다. gate artifact에 `finite=true`, `nonzero=true`가 모두 있어야 IG KD smoke training을 허용한다.

**커밋 메시지:** `FEAT: Add differentiable IG distillation loss`

**설계 근거:** Captum 호출 결과를 그대로 시각화하는 것만으로는 설명 KD가 아니다. 별도 gradient gate가 graph 단절을 본 실험 전에 탐지한다.

---

### Step 7 `[Claude]` `[✓]`: Layer-wise Relevance Propagation KD 구현

> **구현 포인트:** 구현 전 BERT의 residual, LayerNorm, attention에 적용할 LRP propagation rule을 문서화하고 보존성 오차 허용치를 테스트로 고정한다. IG 코드를 LRP 이름으로 재사용하거나 결과 경로를 공유하지 않는다.

**Files:**

- Modify: `src/explainable_kd/xai/lrp.py`
- Modify: `src/explainable_kd/training/kd.py`
- Create: `docs/lrp_rules.md`
- Create: `tests/test_lrp.py`

**Interfaces:**

- Produces: `layerwise_relevance(model, batch, target_class, create_graph) -> AttributionBatch`, `lrp_explanation_loss(teacher_attr, student_attr, valid_mask)`, `verify_lrp_gradient_path(...) -> GradientCheck`
- Loss contract: `total_loss = alpha * task_loss + beta * kd_loss + gamma * lrp_loss`

- [ ] 선택한 epsilon/alpha-beta propagation rule과 BERT 모듈별 처리 방식을 `docs/lrp_rules.md`에 명시한다.
- [ ] 작은 고정 모델에서 output relevance와 input relevance의 보존성 오차가 정한 tolerance 이내인지 검사한다.
- [ ] special token/padding mask, same-target enforcement, normalization을 IG와 동일한 공통 attribution schema로 검사한다.
- [ ] one-batch backward 후 Student gradient가 finite·nonzero이고 Teacher gradient가 없음을 검사한다.
- [ ] LRP gradient gate artifact가 없으면 LRP KD 본 학습을 거부한다.
- [ ] LRP 결과가 `attributions/lrp/...`에만 저장되고 IG 파일을 읽거나 덮어쓰지 않는지 검사한다.

**검증 방법:** `verify-xai-gradients --method lrp --config configs/smoke.yaml`의 보존성, finite, nonzero 결과가 모두 성공한 뒤 depth별 smoke training을 수행한다.

**커밋 메시지:** `FEAT: Add differentiable LRP distillation loss`

**설계 근거:** BERT LRP는 일반 gradient attribution보다 propagation rule 선택 영향이 크다. 규칙과 tolerance를 먼저 고정해야 결과를 LRP라고 재현 가능하게 설명할 수 있다.

---

### Step 8 `[Claude]` `[✓]`: 단일 CLI와 Colab 재개 흐름 구현

> **구현 포인트:** notebook은 명령 조립만 담당한다. CLI가 device와 resolved paths를 항상 출력하고, 실제 train/evaluate/XAI 명령은 CUDA가 아니면 명시적으로 중단한다. 데이터 smoke 검사는 CPU 허용이다.

**Files:**

- Modify: `src/explainable_kd/cli.py`
- Modify: `notebooks/run_experiments_colab.ipynb`
- Create: `tests/test_cli.py`

**Interfaces:**

- Produces commands: `list-experiments`, `prepare-data`, `train`, `train-group`, `verify-xai-gradients`, `evaluate`, `summarize`, `visualize`
- Common args: `--config`, optional `--overlay`, `--artifact-root`, `--seed`, `--resume/--no-resume`

- [ ] argparse subcommands와 canonical experiment/group validation 테스트를 작성한다.
- [ ] `train-group` 순서를 Teacher 존재 확인 → baseline → KD → IG gate/IG KD → LRP gate/LRP KD로 제한한다.
- [ ] `--resume`은 동일 run의 `last/`만 사용하고 완료 artifact가 있는 run은 명시적 `--rerun` 없이는 건너뛰게 한다.
- [ ] 명령 시작 시 device, experiment ID, seed, config hash, data fingerprint, artifact root를 출력한다.
- [ ] notebook setup cell, device check, optional Drive mount, 단계별 CLI cell을 분리하고 Python 로직을 notebook에서 제거한다.
- [ ] CLI parser 테스트는 CPU에서, 실제 CUDA guard와 resume smoke는 Colab에서 검증한다.

**검증 방법:** `python -m explainable_kd.cli --help`와 각 subcommand `--help`가 성공해야 한다. Colab에서 중단한 2-step run을 재실행했을 때 `last/`의 global step 다음부터 이어져야 한다.

**커밋 메시지:** `FEAT: Add resumable Colab experiment CLI`

**설계 근거:** notebook에 상태와 로직을 쌓으면 셀 실행 순서에 따라 결과가 달라진다. 얇은 runner와 파일 기반 checkpoint가 Colab 세션 단절을 가장 단순하게 견딘다.

---

### Step 9 `[Claude]` `[✓]`: 공통 평가·집계·시각화 구현

> **구현 포인트:** 시각화는 모델을 다시 실행하지 않고 저장된 metrics/attribution만 읽는다. latency는 warm-up과 timed iterations를 분리하고 batch size, device, precision을 함께 기록한다.

**Files:**

- Modify: `src/explainable_kd/evaluation/evaluate.py`
- Modify: `src/explainable_kd/evaluation/visualize.py`
- Create: `tests/test_evaluation.py`
- Create: `tests/test_visualization.py`

**Interfaces:**

- Produces: `evaluate_checkpoint(...) -> RunMetrics`, `measure_inference_latency(...) -> EfficiencyMetrics`, `aggregate_runs(artifact_root) -> DataFrame`, `render_report_tables(...)`, `render_token_importance(...)`

- [ ] classification metrics와 parameter count/model size fixture 테스트를 작성한다.
- [ ] CUDA synchronize, warm-up, 반복 횟수, batch size를 고정한 classification-only latency 측정을 구현한다.
- [ ] IG와 LRP 각각 cosine similarity, Spearman correlation, top-k overlap을 valid tokens에서 계산한다.
- [ ] summary가 13개 canonical ID와 요청 seed를 모두 포함하는지 검사하고 누락·중복 run이 있으면 실패시킨다.
- [ ] `experiment_summary.csv/json`에 config, seed, 성능, 효율, 설명 유사도, artifact paths를 저장한다.
- [ ] 동일 case ID에 대해 Teacher, baseline, KD, IG KD, LRP KD의 예측·확률·token importance를 비교 가능한 schema로 만든다.
- [ ] 발표 계획에 맞춘 depth별 성능, efficiency, explanation similarity, trade-off, token heatmap 파일을 생성한다.
- [ ] 작은 fixture artifact만으로 모든 표와 그림이 생성되는 테스트를 작성한다.

**검증 방법:** `pytest tests/test_evaluation.py tests/test_visualization.py -v`가 모델 다운로드 없이 통과하고, fixture에서 기대한 행 수와 figure 파일을 생성해야 한다.

**커밋 메시지:** `FEAT: Add experiment evaluation and visual reports`

**설계 근거:** 평가와 표현을 분리하면 그래프 스타일 수정 때문에 비싼 모델 추론을 반복하지 않는다. summary completeness 검사는 일부 실험만으로 결론을 내리는 오류를 막는다.

---

### Step 10 `[직접]` `[✓]`: 전체 smoke 검증 후 Colab 본 실험 실행

> **구현 포인트:** smoke 결과와 본 실험 결과의 artifact root를 분리한다. 13개 조건을 한 세션에서 무리하게 연속 실행하지 않고 checkpoint 기반으로 단계별 재개한다.

**Files:**

- Create during execution: `<smoke_artifact_root>/...`
- Create during execution: `<persistent_artifact_root>/...`
- Modify after measured incompatibility only: `requirements.txt`

**Interfaces:**

- Consumes: Step 1~9 코드, `configs/base.yaml`, `configs/smoke.yaml`, Colab GPU
- Produces: README의 artifact tree 전체와 13-row-per-seed summary

- [ ] CPU 단위 테스트 전체를 `pytest -q`로 실행한다.
- [ ] Colab device가 `cuda`인지 확인하고 package/version/device 정보를 run metadata에 저장한다.
- [ ] smoke artifact root에서 데이터 준비, Teacher, 3 baseline, 3 KD를 각 2 step 실행한다.
- [ ] IG와 LRP gradient gate를 각각 통과한 뒤 각 설명 KD 3개를 2 step 실행한다.
- [ ] smoke summary가 정확히 13개 조건이며 경로 충돌과 missing loss field가 없는지 검사한다.
- [ ] 본 artifact root에서 Teacher를 먼저 완료하고 best checkpoint hash를 기록한다.
- [ ] baseline → KD → IG KD → LRP KD 순으로 depth 10/8/6을 실행하되 세션 단절 시 `last/`에서 재개한다.
- [ ] 13개 best checkpoint를 동일 test split과 latency protocol로 평가한다.
- [ ] summary CSV/JSON, case comparison, 발표용 figures를 생성한다.
- [ ] 생성 파일을 다시 읽어 config hash, data fingerprint, Teacher hash, seed 일관성을 최종 검사한다.

**검증 방법:** 다음 조건을 모두 만족해야 본 실험 1개 seed가 완료된 것이다.

```text
summary unique experiment_id count == 13
teacher checkpoint count == 1 per seed
all KD-family teacher checkpoint hashes == teacher_d12_supervised best hash
all run data fingerprints == split_manifest fingerprint
all classification metric fields are finite
all classification latency records exclude attribution time
IG and LRP attribution paths are disjoint
every summary artifact path exists
```

**커밋 메시지:** `CHORE: Record reproducible experiment outputs`

**설계 근거:** smoke 검증은 비용이 큰 XAI 본 학습 전에 인터페이스와 gradient 문제를 발견한다. 본 결과는 실제 artifact가 존재할 때만 보고하며 Colab 계산 한계를 만나면 축소한 설정과 한계를 summary에 명시한다.

---

## 구현 시 결정이 필요한 제한된 선택지

### Student 초기화

- **권장:** `bert-base-cased` pretrained state에서 선택한 encoder layer만 유지하고 분류 head를 새로 초기화한다. 세 depth가 같은 축소 정책을 사용해 비교가 단순하다.
- 대안: Teacher의 균등 간격 layer를 Student에 복사한다. 성능 이점 가능성이 있지만 baseline과 KD 사이 초기화 공정성을 별도로 설계해야 한다.

선택 결과는 모든 Student 조건에 동일하게 적용하고 resolved config에 기록한다.

### LRP 구현

- **권장:** BERT 모듈별 propagation rule을 직접 명시하고 tiny-model 보존성·gradient 테스트로 고정한다.
- 대안: 유지보수되는 라이브러리가 조사 시점에 BERT와 higher-order gradient 요구를 모두 충족하면 해당 라이브러리를 사용한다. 버전과 rule을 정확히 pin한다.

라이브러리가 attribution 시각화만 지원하고 Student 설명 손실로 backward되지 않으면 사용할 수 없다.

## 최종 완료 산출물

- 13개 조건별 `best/`, `last/`, train history
- 동일 split/data fingerprint와 resolved config
- Accuracy, Macro Precision, Macro Recall, Macro F1
- parameter count, model size, classification-only latency
- 적용 가능한 task/KD/IG/LRP loss의 분리 로그
- IG/LRP별 token attribution JSONL과 설명 유사도
- `experiment_summary.csv/json`, 사례 비교표, 발표용 그래프·heatmap
- smoke gradient gate와 본 실험 artifact의 명확한 분리
