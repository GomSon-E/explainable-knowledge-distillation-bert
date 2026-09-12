# Explainable BERT Knowledge Distillation for TREC-6

12층 BERT Teacher의 예측 지식과 토큰 중요도를 10·8·6층 Student에 전달하는
질문 유형 분류 프로젝트다. 모든 실제 학습·평가·설명 산출물은 Colab GPU에서
실행하는 것을 기준으로 한다.

## 데이터

- 데이터셋: `lukasgarbas/trec`의 고정 revision
- 모델·토크나이저: `google-bert/bert-base-cased`
- 태스크: TREC-6 6-class 질문 유형 분류
- 클래스: `ABBR`, `ENTY`, `DESC`, `HUM`, `LOC`, `NUM`
- 입력 길이: BERT tokenizer 기준 최대 512 tokens
- 512 tokens 초과 샘플: 제외
- seed 42 기준 split: train 4,906 / validation 546 / test 500
- 전처리: 동일 tokenizer, split, seed를 모든 실험 조건에서 공유

현재 고정 데이터 기준 실제 token 길이 20 이하 샘플은 train 4,473개,
validation 505개, test 497개다.

## 실험 조건

- **① 12층 Teacher:** 실제 정답으로 학습
- **② 10층 대조군:** 실제 정답으로 학습
- **③ 10층 Student:** 실제 정답 + 일반 KD
- **④ 10층 Student:** 실제 정답 + 일반 KD + IG 설명 모방
- **⑤ 10층 Student:** 실제 정답 + 일반 KD + LRP 설명 모방
- **⑥ 8층 대조군:** 실제 정답으로 학습
- **⑦ 8층 Student:** 실제 정답 + 일반 KD
- **⑧ 8층 Student:** 실제 정답 + 일반 KD + IG 설명 모방
- **⑨ 8층 Student:** 실제 정답 + 일반 KD + LRP 설명 모방
- **⑩ 6층 대조군:** 실제 정답으로 학습
- **⑪ 6층 Student:** 실제 정답 + 일반 KD
- **⑫ 6층 Student:** 실제 정답 + 일반 KD + IG 설명 모방
- **⑬ 6층 Student:** 실제 정답 + 일반 KD + LRP 설명 모방

| 조건 | 층 | 학습 방식 |
|---|---:|---|
| `teacher_d12_supervised` | 12 | 실제 정답 라벨 |
| `student_d10_baseline` / `student_d8_baseline` / `student_d6_baseline` | 10 / 8 / 6 | 실제 정답 라벨 |
| `student_d10_kd` / `student_d8_kd` / `student_d6_kd` | 10 / 8 / 6 | task loss + 일반 KD loss |
| `student_d10_ig_kd` / `student_d8_ig_kd` / `student_d6_ig_kd` | 10 / 8 / 6 | task + KD + IG explanation loss |
| `student_d10_lrp_kd` / `student_d8_lrp_kd` / `student_d6_lrp_kd` | 10 / 8 / 6 | task + KD + LRP explanation loss |

모든 KD 조건은 동일한 `teacher_d12_supervised` checkpoint에서 직접 증류한다.
Student 간 순차 증류는 사용하지 않는다. Teacher는 Student 학습 중 freeze한다.

## 구현된 기능

- 12층 Teacher supervised fine-tuning
- 10/8/6층 labels-only baseline fine-tuning
- temperature와 loss weight를 설정할 수 있는 일반 KD
- Integrated Gradients token attribution 추출·시각화
- IG explanation loss를 포함한 10/8/6층 IG KD
- BERT-compatible epsilon relevance 기반 LRP 추출·시각화
- LRP explanation loss를 포함한 10/8/6층 LRP KD
- validation Macro F1 기준 `best` checkpoint
- epoch별 loss·classification metric 저장
- `last` checkpoint 기반 resume
- test Accuracy, Precision, Recall, Macro F1 저장
- parameter count, model size, classification-only inference latency 저장
- IG/LRP Student gradient path tiny-batch 검증

## 아직 구현하지 않은 기능

- 13개 checkpoint를 자동으로 모아 평가하는 `evaluate --all`
- 전체 실험 summary CSV/JSON 생성
- summary 기반 발표용 그래프 자동 생성

## 주요 코드

```text
src/explainable_kd/
├── cli.py                    # 모든 실행 명령의 진입점
├── data/pipeline.py          # TREC-6 로딩·split·tokenizer·512 길이 필터
├── common/config.py          # YAML 설정과 loss/XAI 설정
├── common/checkpoint.py      # checkpoint·metric 경로
├── training/teacher.py       # 12층 Teacher
├── training/baseline.py      # 10/8/6층 baseline
├── training/kd.py            # 일반 KD
├── training/ig_kd.py         # IG explanation KD
├── training/lrp_kd.py        # LRP explanation KD
├── xai/ig.py                # IG 추출·유사도·시각화
└── xai/lrp.py               # LRP 추출·유사도·시각화
```

## 실행

설치와 GPU 확인:

```bash
pip install -r requirements.txt
pip install .
python -m explainable_kd.cli device
```

Teacher smoke:

```bash
python -m explainable_kd.cli train-teacher \
  --config configs/base.yaml \
  --overlay configs/smoke.yaml \
  --artifact-root /content/drive/MyDrive/hanyang/artifacts-smoke
```

Baseline, 일반 KD:

```bash
python -m explainable_kd.cli train-baselines \
  --config configs/base.yaml \
  --artifact-root /content/drive/MyDrive/hanyang/artifacts

python -m explainable_kd.cli train-kd \
  --config configs/base.yaml \
  --artifact-root /content/drive/MyDrive/hanyang/artifacts
```

IG attribution 및 IG KD:

```bash
python -m explainable_kd.cli extract-ig \
  --config configs/base.yaml \
  --overlay configs/smoke.yaml \
  --artifact-root /content/drive/MyDrive/hanyang/artifacts-smoke \
  --max-examples 1

python -m explainable_kd.cli train-ig-kd \
  --config configs/base.yaml \
  --artifact-root /content/drive/MyDrive/hanyang/artifacts
```

LRP attribution 및 LRP KD:

```bash
python -m explainable_kd.cli extract-lrp \
  --config configs/base.yaml \
  --overlay configs/smoke.yaml \
  --artifact-root /content/drive/MyDrive/hanyang/artifacts-smoke \
  --max-examples 1

python -m explainable_kd.cli train-lrp-kd \
  --config configs/base.yaml \
  --artifact-root /content/drive/MyDrive/hanyang/artifacts
```

`train-kd`, `train-ig-kd`, `train-lrp-kd`는 먼저 같은 artifact root의
`teacher_d12_supervised/seed_42/best`를 읽는다. 따라서 Teacher 학습을 먼저
완료해야 한다. `configs/smoke.yaml`은 train 32개, validation 16개, test 16개와
1 epoch·2 optimizer steps를 사용한다.

## 산출물

실행 결과는 `--artifact-root` 아래에 저장한다.

```text
artifacts/
├── checkpoints/<experiment_id>/seed_<seed>/{best,last}/
├── metrics/<experiment_id>/seed_<seed>/metrics.json
├── metrics/<experiment_id>/seed_<seed>/metrics.csv
├── metrics/ig_similarity/seed_<seed>.json
├── metrics/lrp_similarity/seed_<seed>.json
├── attributions/ig/<experiment_id>/seed_<seed>/examples.jsonl
├── attributions/lrp/<experiment_id>/seed_<seed>/examples.jsonl
└── figures/token_importance/{ig,lrp}/*.png
```

Attribution row에는 `example_id`, `tokens`, `scores`, `valid_mask`,
`target_class`, `predicted_class`가 저장된다. special token과 padding은
`valid_mask=false`이며, 유사도와 그림에서는 제외한다.

## 개발 검증

```bash
pytest -q
```

현재 테스트 결과는 `37 passed, 1 skipped`다. 실제 GPU 학습 결과는 Colab에서
생성된 checkpoint와 metrics가 존재할 때만 실험 결과로 간주한다.

Colab 실행 notebook은 [`notebooks/run_experiments_colab.ipynb`](notebooks/run_experiments_colab.ipynb)다.
