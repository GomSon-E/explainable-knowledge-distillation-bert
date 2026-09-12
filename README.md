# Explainable BERT Knowledge Distillation for TREC-6

이 저장소는 12층 BERT Teacher의 예측 분포와 토큰 중요도를 10·8·6층
Student에 전달하는 13개 실험을 Google Colab GPU에서 재현하기 위한
프로젝트다. 현재 구현 범위는 공통 데이터 기반, 12층 Teacher fine-tuning,
10·8·6층 labels-only baseline, 일반 KD, IG attribution 추출·시각화와 smoke
test다. IG 설명 손실 학습과 LRP는 아직 구현하지 않는다.

설계 기준 문서는 `AGENTS.md`, `01_video_notes.md`,
`02_assignment_brief.md`, `03_experiment_plan.md`,
`04_presentation_plan.md`이다.

## 구조

```text
.
├── configs/
│   ├── base.yaml                 # 모든 조건이 공유하는 데이터·재현성·평가 설정
│   ├── experiments.yaml          # 13개 조건의 유일한 ID와 Teacher 참조
│   └── smoke.yaml                # 소량 데이터·소수 step·XAI 1배치 검증 override
├── notebooks/
│   └── run_experiments_colab.ipynb
│                                  # 설치, 장치 확인, Drive 연결, CLI 호출만 담당
├── src/explainable_kd/
│   ├── cli.py                    # 향후 모든 Colab 실행 명령의 단일 진입점
│   ├── data/pipeline.py          # TREC-6 분할·길이 정책·토큰화·manifest
│   ├── common/
│   │   ├── config.py             # 설정 병합·검증·experiment registry
│   │   ├── seed.py               # Python/NumPy/PyTorch/DataLoader seed
│   │   ├── metrics.py            # 분류·효율·설명 유사도 지표
│   │   └── checkpoint.py         # best/last 저장과 세션 재개
│   ├── training/
│   │   ├── teacher.py            # 12층 supervised Teacher
│   │   ├── baseline.py           # 10·8·6층 labels-only baseline
│   │   └── kd.py                 # 일반 KD와 IG/LRP 손실 결합 orchestration
│   ├── xai/
│   │   ├── ig.py                 # 미분 가능한 IG와 토큰 정렬
│   │   └── lrp.py                # 미분 가능한 LRP와 토큰 정렬
│   └── evaluation/
│       ├── evaluate.py           # 체크포인트 평가와 효율 측정
│       └── visualize.py          # 요약표·그래프·토큰 중요도 그림
├── artifacts/                    # 생성물 루트; 아래 저장 규칙 준수
├── tests/                        # 향후 단위·smoke·gradient-path 검사
├── requirements.txt              # Colab 설치 목록
└── experiment_pipeline_plan.md   # 단계별 구현 TODO와 검증 계획
```

빈 Python 모듈에는 책임 범위와 구현 TODO만 기록되어 있다. notebook에
학습 로직을 넣지 않고, 향후 `python -m explainable_kd.cli ...` 호출만 두는
구조를 유지한다.

## 13개 실험 조건

`configs/experiments.yaml`이 실험 ID의 단일 원본이다. ID를 별칭으로 줄이거나
층 수만으로 산출물 디렉터리를 만들지 않는다.

| 순번 | `experiment_id` | 층 | 실제 정답 | Teacher 분포 | 설명 손실 |
|---:|---|---:|:---:|:---:|:---:|
| 1 | `teacher_d12_supervised` | 12 | O | X | X |
| 2 | `student_d10_baseline` | 10 | O | X | X |
| 3 | `student_d10_kd` | 10 | O | O | X |
| 4 | `student_d10_ig_kd` | 10 | O | O | IG |
| 5 | `student_d10_lrp_kd` | 10 | O | O | LRP |
| 6 | `student_d8_baseline` | 8 | O | X | X |
| 7 | `student_d8_kd` | 8 | O | O | X |
| 8 | `student_d8_ig_kd` | 8 | O | O | IG |
| 9 | `student_d8_lrp_kd` | 8 | O | O | LRP |
| 10 | `student_d6_baseline` | 6 | O | X | X |
| 11 | `student_d6_kd` | 6 | O | O | X |
| 12 | `student_d6_ig_kd` | 6 | O | O | IG |
| 13 | `student_d6_lrp_kd` | 6 | O | O | LRP |

모든 KD 계열의 `teacher_ref`는 반드시 `teacher_d12_supervised`이다.
Baseline은 Teacher 체크포인트를 읽지 않는다. Student 간 순차 증류는 금지한다.

## Artifact 저장 규칙

## LRP 규칙과 구현 범위

`src/explainable_kd/xai/lrp.py`는 BERT 입력 embedding에서 target logit까지의
epsilon-stabilized relevance를 계산한다. 현재 Hugging Face BERT 내부 graph를
모든 연산 단위로 분해하는 표준 Captum 경로가 없으므로, 다음의 명시적
BERT-compatible 규칙을 사용한다.

| 구성요소 | relevance 처리 |
|---|---|
| Linear·embedding | `x * d(logit)/dx`를 epsilon으로 안정화·정규화 |
| Attention | attention weight 자체를 증거로 취급하지 않고 token/value 경로로 보존 |
| Residual | shortcut과 변환 branch의 입력 contribution으로 relevance 보존 |
| LayerNorm | 평균·분산 통계로 relevance를 만들지 않고 입력 방향으로 보존 |
| GELU·dropout | 입력 contribution 방향으로 전달 |

이 범위는 gradient×input을 LRP로 이름만 바꾼 것이 아니라, 위 규칙과
`LRP_RULES` 메타데이터를 함께 저장하는 epsilon relevance 구현이다. 다만
attention/residual/LayerNorm의 exact operator-level 분해가 필요하면 별도의
BERT graph tracer가 필요하며, Colab smoke와 본 실험 결과에는 이 제한을
명시한다.

실행 한 건의 식별자는 `(experiment_id, seed)`이다. 체크포인트와 run별
결과는 항상 다음 형식을 사용한다.

```text
artifacts/
├── data/
│   ├── split_manifest.json
│   └── processed/<data_fingerprint>/...
├── checkpoints/<experiment_id>/seed_<seed>/
│   ├── best/                     # validation 기준 최적 모델
│   └── last/                     # Colab 중단 복구용 최신 상태
├── logs/<experiment_id>/seed_<seed>/train_history.jsonl
├── metrics/<experiment_id>/seed_<seed>/
│   ├── classification.json
│   ├── efficiency.json
│   └── attribution_similarity.json
├── attributions/<xai_method>/<experiment_id>/seed_<seed>/examples.jsonl
├── reports/
│   ├── experiment_summary.csv
│   ├── experiment_summary.json
│   └── case_comparison.csv
└── figures/
    ├── performance_by_depth.png
    ├── efficiency_tradeoff.png
    ├── attribution_similarity.png
    └── token_importance/<case_id>_<xai_method>.png
```

저장 계약은 다음과 같다.

- `<experiment_id>`는 registry의 13개 값 중 하나만 허용한다.
- `<xai_method>`는 `ig` 또는 `lrp`만 허용한다. 두 결과를 같은 파일에
  덮어쓰지 않는다.
- 모든 JSON/JSONL row에는 `experiment_id`, `seed`, `config_hash`,
  `data_fingerprint`, `created_at_utc`, `artifact_paths`를 기록한다.
- 학습 로그에는 적용 가능한 `task_loss`, `kd_loss`, `ig_loss`, `lrp_loss`,
  `total_loss`를 별도 필드로 둔다. 적용되지 않는 손실은 `0`으로 위장하지
  않고 `null`로 저장한다.
- 토큰 attribution row에는 `example_id`, `tokens`, `scores`, `valid_mask`,
  `target_class`, `predicted_class`를 저장한다. 특수 토큰과 padding은
  `valid_mask=false`로 명시한다.
- 분류 latency에는 IG/LRP 계산 시간을 포함하지 않는다. 필요 시 설명 계산
  시간은 별도의 `attribution_latency_ms` 필드로 저장한다.
- `best/`와 `last/`는 같은 run 디렉터리 안에서만 갱신하며 다른 seed나
  실험 ID의 체크포인트를 자동 탐색하지 않는다.

## 데이터·공통 기반 실행

`configs/base.yaml`은 `lukasgarbas/trec`의 고정 revision을 사용한다. 이
Parquet 기반 TREC-6 mirror는 `datasets==4.1.1`에서 Python dataset script
오류 없이 로드되며, 원본 train/validation 5,452개를 다시 합쳐 seed 42로
90/10 validation을 만들고 원본 test 500개는 유지한다. coarse label은 다음
순서로 정수 매핑한다.

`ABBR=0`, `ENTY=1`, `DESC=2`, `HUM=3`, `LOC=4`, `NUM=5`

Colab에서 저장소 루트로 이동한 뒤 다음 순서로 실행한다.

```bash
pip install -r requirements.txt
pip install .
python -m explainable_kd.cli device
python -m explainable_kd.cli prepare-data \
  --config configs/base.yaml \
  --overlay configs/smoke.yaml \
  --artifact-root /content/hanyang-artifacts
python -m explainable_kd.cli smoke-test \
  --config configs/base.yaml \
  --overlay configs/smoke.yaml \
  --artifact-root /content/hanyang-artifacts \
  --batch-size 4
```

`device`는 `CUDA available`과 선택된 device를 출력한다. `prepare-data`는
`data/split_manifest.json`과 `data/processed/<fingerprint>/`를 만들고,
`smoke-test`는 `metrics/smoke_test.json`에 실제 batch tensor shape·dtype를
기록한다. Google Drive를 사용하려면 `/content/drive/...`를
`--artifact-root`에 전달한다.

### Smoke test 결과

2026-09-12에 고정 revision과 실제 BERT tokenizer로 로컬 CPU에서 위와 같은
CLI 경로를 실행했다. 이는 Colab GPU 학습 결과가 아니라 데이터 입력 계약을
검증한 결과다.

```text
status: passed
split_counts: train=32, validation=16, test=16
batch_size: 4
sequence_length: 10
input_ids: [4, 10] torch.int64
attention_mask: [4, 10] torch.int64
token_type_ids: [4, 10] torch.int64
CUDA available: False
Device: cpu
```

Colab에서는 같은 명령에서 `CUDA available: True`, `Device: cuda`가
출력되는지 먼저 확인한다. smoke artifact JSON에는 dataset/tokenizer
revision, data fingerprint, split counts, tensor shapes가 함께 저장된다.

## Colab 실행 준비

1. Colab에서 GPU runtime을 선택한다.
2. 저장소를 `/content` 아래 clone 또는 upload한다.
3. 영속 저장이 필요하면 Google Drive를 mount하고 artifact root만 Drive
   경로로 override한다. 코드나 데이터 경로에 로컬 VS Code 절대 경로를 쓰지 않는다.
4. `notebooks/run_experiments_colab.ipynb`의 setup과 device check 셀을 실행한다.
5. 저장소 루트에서 `pip install -r requirements.txt`와 `pip install .`을
   실행한다. 구형 로컬 pip의 editable 설치는 사용하지 않는다.

notebook에는 데이터 준비·smoke·Teacher 학습 명령이 활성화되어 있으며,
baseline·일반 KD와 전체 학습 명령은 주석으로 제공한다.

## Colab 실행 순서와 단계별 I/O

| 단계 | 향후 CLI 동작 | 입력 | 출력·체크포인트 | 진행 조건 |
|---:|---|---|---|---|
| 0 | setup/device check | `requirements.txt`, Colab runtime | PyTorch device 출력, artifact root 확인 | 실제 실험은 `cuda` 확인 후 진행 |
| 1 | `prepare-data` | `configs/base.yaml`, TREC-6 | `split_manifest.json`, `processed/<fingerprint>/` | 모든 조건이 같은 fingerprint 사용 |
| 2 | `train --experiment teacher_d12_supervised` | processed data, seed | Teacher `best/`, `last/`, train log | Teacher best checkpoint 존재 |
| 3 | baseline 3개 학습 | data, 각 depth config | 각 baseline checkpoint/log | Teacher를 로드하지 않았음을 config에 기록 |
| 4 | KD 3개 학습 | data, 고정 Teacher best | 각 KD checkpoint/log | Teacher frozen 검증 |
| 5 | `verify-xai-gradients --method ig` | smoke data, Teacher, 임시 Student | gradient 검사 JSON | Student gradient가 finite·nonzero |
| 6 | IG KD 3개 학습 | data, 고정 Teacher, IG 설정 | 각 IG KD checkpoint/log | 같은 target class·token mask 사용 |
| 7 | `verify-xai-gradients --method lrp` | smoke data, Teacher, 임시 Student | gradient 검사 JSON | Student gradient가 finite·nonzero |
| 8 | LRP KD 3개 학습 | data, 고정 Teacher, LRP 설정 | 각 LRP KD checkpoint/log | IG 산출물과 경로 분리 |
| 9 | `evaluate --all` | 13개 best checkpoint, test split | run별 metrics, attribution JSONL | 분류 latency에서 XAI 제외 |
| 10 | `summarize` | run별 metrics와 config | summary CSV/JSON, 사례 비교표 | 13개 조건 누락 검사 |
| 11 | `visualize` | summary와 attribution JSONL | 발표용 표·그래프·heatmap | 저장된 데이터만 사용 |

실제 본 실험 전에 `configs/smoke.yaml`을 overlay하여 데이터 경로, 체크포인트
재개, loss logging, IG/LRP gradient 연결을 최소 비용으로 검증한다.

## 완료 판단

터미널 출력만으로 실험 성공을 주장하지 않는다. 해당 run의 checkpoint,
metrics JSON, config hash가 실제로 존재하고 summary에서 참조될 때만 완료로
간주한다. 자세한 구현 순서와 테스트 기준은 `experiment_pipeline_plan.md`를
따른다.
