# 구현 TODO와 단계별 Codex 프롬프트

## 사용 방식

- 각 단계를 시작할 때마다 아래 공통 문장을 프롬프트 첫 줄에 붙인다.
- 앞 단계의 Colab 실행 결과를 확인한 뒤 다음 단계로 진행한다.
- 코드 작성은 VS Code/Codex에서 하지만, 실제 학습·평가·XAI 계산은 Google Colab GPU에서 실행한다.

### 공통 첫 문장

```text
프로젝트 루트의 AGENTS.md와 01~04 markdown 문서를 먼저 읽고, 그 규칙을 준수해줘.
```

---

## 0. 구현 계획·폴더 구조 설계

### 완료 기준

- 코드 뼈대와 Colab 실행 순서가 정리되어 있다.
- 아직 모델 학습은 수행하지 않는다.

### Codex 프롬프트

```text
이번 프로젝트의 구현 계획과 Colab 실행 구조를 먼저 설계해줘. 아직 모델 학습 코드는 구현하지 마.

다음이 포함된 폴더 구조와 TODO를 만들어줘.
- 데이터 처리
- 공통 config·seed·metrics·checkpoint 유틸
- Teacher 학습
- 10·8·6층 baseline
- 일반 KD
- IG KD
- LRP KD
- 전체 평가와 결과 시각화
- Colab에서 실행할 notebook 또는 실행 진입점

각 파일의 역할, Colab에서 실행하는 순서, 각 단계의 입력·출력 체크포인트를 README에 정리해줘.
```

---

## 1. 공통 기반: 데이터·평가·Colab 환경

### 완료 기준

- TREC-6를 불러오고, 한 배치가 BERT 입력 형태까지 정상적으로 변환된다.
- metric 계산과 Colab GPU 확인이 가능하다.

### Codex 프롬프트

```text
공통 데이터·실험 기반을 구현해줘.

- TREC-6 영어 질문 유형 분류 데이터셋을 로드하고 6개 클래스를 일관되게 매핑할 것
- BERT base cased tokenizer를 사용할 것
- train/validation/test split, random seed, config 관리 구현
- Accuracy, Precision, Recall, Macro F1 계산 구현
- checkpoint, metrics JSON/CSV, plot 저장 경로를 Colab 환경에서 설정 가능하게 만들 것
- CUDA 사용 가능 여부와 device를 출력할 것
- Colab에서 작은 샘플 배치를 실제로 tokenization → BERT 입력 형태까지 통과하는 smoke test를 만들 것

아직 Teacher 학습이나 Student/KD/XAI는 구현하지 마.
실행 방법과 smoke test 결과를 README에 적어줘.
```

---

## 2. 12층 Teacher 학습

### 완료 기준

- 12층 Teacher checkpoint와 test 성능표가 생성된다.

### Codex 프롬프트

```text
12층 BERT Teacher fine-tuning을 구현해줘.

- google-bert/bert-base-cased를 TREC-6 6-class 분류 모델로 fine-tuning할 것
- validation Macro F1 기준 best checkpoint를 저장할 것
- 재개 학습(resume) 가능하게 만들 것
- epoch별 train/validation loss와 metric을 저장할 것
- 최종 test Accuracy, Precision, Recall, Macro F1을 저장할 것
- 파라미터 수, 모델 파일 크기, 분류 예측만의 추론 시간을 측정할 것
- Colab GPU에서 smoke test 1 epoch와 전체 학습을 각각 실행할 수 있게 만들 것

아직 Student, KD, IG, LRP는 구현하지 마.
```

---

## 3. 10·8·6층 Baseline

### 완료 기준

- Teacher 없이 실제 정답만으로 학습한 10·8·6층 모델 세 개가 있다.

### Codex 프롬프트

```text
Teacher 코드를 변경하지 말고, 10층·8층·6층 BERT baseline을 구현해줘.

- 12층 BERT에서 Transformer layer 수만 줄인 10/8/6층 모델을 구성할 것
- 각 baseline은 Teacher를 사용하지 않고 실제 정답 라벨만으로 학습할 것
- Teacher와 완전히 동일한 데이터 split, tokenizer, seed 정책, metric, checkpoint 규칙을 사용할 것
- 각 깊이별 best checkpoint, test 성능, 파라미터 수, 모델 크기, 분류 추론 시간을 저장할 것
- 결과 파일 이름에 baseline과 layer 수가 명확히 드러나게 할 것

일반 KD와 XAI KD는 아직 구현하지 마.
```

---

## 4. 일반 KD

### 완료 기준

- 10·8·6층 일반 KD Student 세 개가 학습되고, task/KD loss가 각각 기록된다.

### Codex 프롬프트

```text
일반 Knowledge Distillation을 구현해줘.

- 이미 학습된 고정된 12층 Teacher 하나를 모든 Student의 Teacher로 사용할 것
- Teacher는 Student 학습 중 freeze하고 gradient를 계산하지 않을 것
- 10층·8층·6층 Student 각각에 대해 task loss와 KD loss를 결합해 학습할 것
- temperature와 각 loss 가중치는 config로 분리할 것
- task loss와 KD loss를 epoch별로 별도 저장할 것
- baseline과 같은 방식으로 test 성능, 모델 크기, 추론 시간을 저장할 것
- Teacher → 10층 → 8층처럼 순차 증류하지 말고, 모두 12층 Teacher에서 직접 증류할 것

IG와 LRP는 아직 구현하지 마.
```

---

## 5. IG 설명 추출·시각화

### 완료 기준

- Teacher와 일반 KD Student 하나에서 토큰 중요도 그림이 생성된다.
- 이 단계에서는 아직 IG 설명 손실로 학습하지 않는다.

### Codex 프롬프트

```text
Integrated Gradients 기반 토큰 중요도 추출과 시각화를 구현해줘. 아직 IG 설명 손실로 학습하지 마.

- Teacher와 Student에 대해 동일한 target class 기준으로 IG attribution을 계산할 것
- special token과 padding은 중요도 비교·시각화에서 제외하거나 명확히 처리할 것
- 토큰 문자열, attribution score, 예측 클래스, 실제 클래스를 저장할 것
- Teacher와 baseline/일반 KD Student의 동일 질문에 대한 토큰 중요도 색칠 그림을 만들 것
- Teacher–Student IG 중요도 유사도(cosine similarity 등)를 계산할 것
- Colab GPU에서 작은 샘플로 실제 결과가 생성되는 smoke test를 만들 것
```

---

## 6. IG 설명 KD

### 완료 기준

- 10·8·6층 IG KD 모델 세 개와 설명 손실 로그가 생성된다.

### Codex 프롬프트

```text
IG explanation distillation을 구현해줘.

- 일반 KD의 task loss + KD loss에 Teacher–Student IG attribution 차이를 explanation loss로 추가할 것
- explanation loss가 Student 파라미터까지 미분 가능하게 연결되는지 먼저 tiny batch에서 검증할 것
- Teacher attribution은 고정하고, Student attribution을 통해 Student만 업데이트할 것
- task loss, KD loss, IG explanation loss, total loss를 각각 저장할 것
- 10층·8층·6층 Student에 대해 별도 학습·checkpoint·평가를 수행할 것
- 최종적으로 성능, 효율, Teacher와의 IG 중요도 유사도, 토큰 중요도 시각화를 저장할 것
```

---

## 7. LRP 설명 추출·LRP KD

### 완료 기준

- LRP 조건 세 개가 완성되거나, 구현 제한과 대안이 명확히 기록된다.

### Codex 프롬프트

```text
LRP 기반 토큰 중요도 추출을 먼저 구현하고, Teacher와 Student에서 같은 target class에 대한 LRP 결과가 정상적으로 나오는지 검증해줘.

그 다음 LRP explanation distillation을 구현해줘.
- task loss + KD loss + Teacher–Student LRP explanation loss로 학습할 것
- BERT의 attention, residual connection, layer normalization 처리 방식은 코드와 문서에 명확히 기록할 것
- Student까지 gradient가 전달되는지 tiny batch로 검증할 것
- 10층·8층·6층 LRP KD 조건을 별도로 학습·평가할 것
- 성능, 효율, LRP 중요도 유사도, 시각화 결과를 저장할 것

Colab GPU에서 계산량 또는 구현상 제약이 있으면, 임의로 생략하지 말고 원인·대안·영향을 보고해줘.
```

---

## 8. 13개 조건 통합 평가·발표용 산출물

### 완료 기준

- 발표에 넣을 표·그래프·이미지·대표 사례가 생성된다.

### Codex 프롬프트

```text
13개 실험 조건의 결과를 통합해 발표용 산출물을 생성해줘.

- 조건별 Accuracy, Precision, Recall, Macro F1 표
- 조건별 파라미터 수, 모델 크기, 분류 추론 시간 표
- 층 수별 baseline·일반 KD·IG KD·LRP KD 성능 비교 그래프
- 성능과 파라미터 수 또는 추론 시간의 trade-off 그래프
- Teacher–Student IG/LRP 중요도 유사도 비교 표와 그래프
- 동일 질문에 대한 Teacher, baseline, 일반 KD, IG KD, LRP KD의 단어 중요도 그림
- 성공 사례와 실패 사례의 예측·설명 비교표
- 모든 결과를 하나의 CSV/JSON summary로 병합

결과가 없는 조건은 추정값을 넣지 말고 `not run`으로 명확히 표기해줘.
```

---

## 진행 원칙

- 1~4단계를 완료하면 Teacher·경량화·일반 KD라는 발표의 기본 뼈대가 확보된다.
- 5~7단계는 XAI를 추가하는 단계이며, IG 완료 후 LRP로 진행한다.
- 매 단계에서 Colab 실행 결과와 저장 산출물을 확인한 뒤 다음 단계로 넘어간다.

## 각 단계 완료 후 Colab에서 할 일

아래 절차는 각 단계의 Codex 프롬프트가 끝난 직후 실행한다. 이 프로젝트에는
`scripts/run_experiments_colab.py`가 없으며, Colab 실행 진입점은
`notebooks/run_experiments_colab.ipynb`와 `python -m explainable_kd.cli`이다.

현재 저장소에서 실제로 사용할 수 있는 CLI 명령은 `device`,
`list-experiments`, `prepare-data`, `smoke-test`, `train-teacher`이다. 3번 이후에 적힌
`train`, `train-group`, `verify-xai-gradients`, `evaluate`, `summarize`,
`visualize`는 해당 구현 단계를 완료하면서 CLI에 추가한 뒤 사용한다. 각 단계
완료 후에는 먼저 `!python -m explainable_kd.cli --help`로 명령이 실제 추가됐는지
확인하며, 구현된 인자 이름이 아래 예시와 달라졌다면 `README.md`와 CLI `--help`를
우선한다.

### Colab 최초 준비

- Colab 런타임 유형을 GPU로 변경한다.
- 실제 GitHub 저장소를 `/content` 아래에 clone하고 저장소 루트로 이동한다.

```python
!git clone https://github.com/GomSon-E/explainable-knowledge-distillation-bert.git /content/explainable-knowledge-distillation-bert
%cd /content/explainable-knowledge-distillation-bert
%pip install -r requirements.txt
%pip install .
```

- 세션이 끊겨도 산출물을 유지하려면 Drive를 mount하고 아래처럼 경로만 바꾼다.
  smoke 결과와 본 실험 결과는 서로 다른 폴더에 저장한다.

```python
from google.colab import drive
drive.mount("/content/drive")

SMOKE_ARTIFACT_ROOT = "/content/drive/MyDrive/hanyang/artifacts-smoke"
ARTIFACT_ROOT = "/content/drive/MyDrive/hanyang/artifacts"
GIT_RESULTS_ROOT = "/content/explainable-knowledge-distillation-bert/results"
```

- Drive를 사용하지 않을 때는 다음 경로를 사용한다. 이 경우 Colab 세션이
  종료되면 산출물이 사라질 수 있다.

```python
SMOKE_ARTIFACT_ROOT = "/content/explainable-knowledge-distillation-bert/artifacts-smoke"
ARTIFACT_ROOT = "/content/explainable-knowledge-distillation-bert/artifacts"
```

- 보관 위치를 분리한다. checkpoint와 processed dataset은 `ARTIFACT_ROOT`의
  Google Drive에만 보관하고, Git에는 코드·설정·notebook과 가벼운 결과(metrics,
  figures, reports, attribution examples, split manifest)만 복사한다.

```python
from pathlib import Path
import shutil

GIT_RESULTS_ROOT = "/content/explainable-knowledge-distillation-bert/results"

def copy_lightweight_results(artifact_root):
    source = Path(artifact_root)
    target = Path(GIT_RESULTS_ROOT) / source.name
    for relative in ("metrics", "figures", "reports", "attributions", "data/split_manifest.json"):
        src = source / relative
        dst = target / relative
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    print(f"Copied lightweight results: {source} -> {target}")
```

### 0번 완료 후

- VS Code에서 생성된 폴더 구조와 문서를 명령어로 확인한다.

```bash
find . -maxdepth 2 -type d | sort
test -f README.md && echo "OK README.md" || echo "MISSING README.md"
test -f experiment_pipeline_plan.md && echo "OK experiment_pipeline_plan.md" || echo "MISSING experiment_pipeline_plan.md"
test -f notebooks/run_experiments_colab.ipynb && echo "OK notebook" || echo "MISSING notebook"
test -d src/explainable_kd && echo "OK package" || echo "MISSING package"
```

- Colab 진입점과 Python 패키지 경로를 명령어로 확인한다.

```python
!test -f notebooks/run_experiments_colab.ipynb && echo "OK notebook" || echo "MISSING notebook"
!test -d src/explainable_kd && echo "OK package" || echo "MISSING package"
```
- 아직 Colab에서 학습을 실행하지 않는다.
- 변경사항을 위 GitHub 저장소에 push한 뒤 Colab에서 새로 clone하거나 pull한다.

```python
!git add -A
!git commit -m "CHORE: Sync Implementation Before Colab"
!git push origin master
```

### 1번 완료 후

- 위의 **Colab 최초 준비**를 실행한 뒤 장치와 13개 실험 ID를 명령어로 확인한다.

```python
!python -m explainable_kd.cli device
!python -m explainable_kd.cli list-experiments --registry configs/experiments.yaml
```

```python
import subprocess
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.get_device_name(0), "CUDA device name is unavailable"
print("CUDA OK:", torch.cuda.get_device_name(0))
```

- smoke 전용 경로에 TREC-6 처리 결과를 만들고 실제 BERT 입력 한 배치를 확인한다.

```python
!python -m explainable_kd.cli prepare-data \
  --config configs/base.yaml \
  --overlay configs/smoke.yaml \
  --artifact-root {SMOKE_ARTIFACT_ROOT}
!python -m explainable_kd.cli smoke-test \
  --config configs/base.yaml \
  --overlay configs/smoke.yaml \
  --artifact-root {SMOKE_ARTIFACT_ROOT} \
  --batch-size 4
```

- 출력에서 `CUDA available: True`, `Device: cuda`를 명령어로 확인한다.

```python
import torch

assert torch.cuda.is_available()
print("CUDA available: True")
print("Device: cuda")
```

- 다음 smoke artifact가 생성됐는지 명령어로 확인한다.

```python
from pathlib import Path

smoke_root = Path(SMOKE_ARTIFACT_ROOT)
paths = [
    smoke_root / "data/split_manifest.json",
    smoke_root / "metrics/smoke_test.json",
]
for path in paths:
    print("OK" if path.exists() else "MISSING", path)
processed = list((smoke_root / "data/processed").glob("*"))
print("OK" if processed else "MISSING", smoke_root / "data/processed/<data_fingerprint>")
```

### 2번 완료 후

- Colab 저장소 루트에서 최신 변경사항을 먼저 반영한다.

```python
%cd /content/explainable-knowledge-distillation-bert
!git pull --ff-only
%pip install -e .
```

- `train-teacher` 명령이 CLI에 추가됐는지 확인하고, smoke 설정으로 Teacher를 먼저
  학습한다.

```python
!python -m explainable_kd.cli --help
!python -m explainable_kd.cli train-teacher \
  --config configs/base.yaml \
  --overlay configs/smoke.yaml \
  --artifact-root {SMOKE_ARTIFACT_ROOT}
```

- smoke checkpoint와 metrics를 확인한 뒤, overlay만 제거하여 Drive의 본 실험
  경로에서 전체 Teacher를 학습한다.

```python
!python -m explainable_kd.cli train-teacher \
  --config configs/base.yaml \
  --artifact-root {ARTIFACT_ROOT}
```

- `{ARTIFACT_ROOT}/checkpoints/teacher_d12_supervised/seed_42/{best,last}/`,
  `{ARTIFACT_ROOT}/metrics/teacher_d12_supervised/seed_42/metrics.json`,
  `{ARTIFACT_ROOT}/metrics/teacher_d12_supervised/seed_42/metrics.csv`가
  실제로 저장됐는지 확인한다.

```python
from pathlib import Path

paths = [
    Path(ARTIFACT_ROOT) / "checkpoints/teacher_d12_supervised/seed_42/best",
    Path(ARTIFACT_ROOT) / "checkpoints/teacher_d12_supervised/seed_42/last",
    Path(ARTIFACT_ROOT) / "metrics/teacher_d12_supervised/seed_42/metrics.json",
    Path(ARTIFACT_ROOT) / "metrics/teacher_d12_supervised/seed_42/metrics.csv",
]
for path in paths:
    print("OK" if path.exists() else "MISSING", path)
```

- `metrics.json`의 `history`, `test`, `efficiency`에 epoch 기록, 최종 분류 지표,
  파라미터 수·모델 파일 크기·분류 전용 추론 시간이 들어있는지 확인한다.

```python
import json

with open(f"{ARTIFACT_ROOT}/metrics/teacher_d12_supervised/seed_42/metrics.json") as f:
    metrics = json.load(f)
for key in ("history", "test", "efficiency"):
    print("OK" if key in metrics else "MISSING", key)
for key in ("parameter_count", "model_file_size_mb", "classification_inference_ms_per_batch"):
    print("OK" if key in metrics["efficiency"] else "MISSING", key)
```

- smoke 및 전체 학습 결과를 저장소에 commit한다.

```python
copy_lightweight_results(SMOKE_ARTIFACT_ROOT)
copy_lightweight_results(ARTIFACT_ROOT)
!git add configs src notebooks results
!git commit -m "DOCS: Add Teacher Training Results"
```

### 3번 완료 후

- 최신 코드와 설정을 먼저 받는다.

```python
%cd /content/explainable-knowledge-distillation-bert
!git pull --ff-only
```

```python
!python -m explainable_kd.cli train-baselines \
  --config configs/base.yaml \
  --overlay configs/smoke.yaml \
  --artifact-root {SMOKE_ARTIFACT_ROOT}
```

- `student_d10_baseline`, `student_d8_baseline`, `student_d6_baseline`의 독립
  checkpoint와 로그가 생성되고 Teacher checkpoint를 읽지 않았는지 확인한다.

```python
from pathlib import Path

for experiment_id in ("student_d10_baseline", "student_d8_baseline", "student_d6_baseline"):
    root = Path(SMOKE_ARTIFACT_ROOT) / "checkpoints" / experiment_id / "seed_42"
    print(experiment_id, "best", (root / "best").exists(), "last", (root / "last").exists())
    print("metrics", (Path(SMOKE_ARTIFACT_ROOT) / "metrics" / experiment_id / "seed_42/metrics.json").exists())
```
- smoke 성공 후 같은 명령에서 `--overlay configs/smoke.yaml`을 제거하고
  `--artifact-root {ARTIFACT_ROOT}`로 바꿔 본 학습을 실행한다.

```python
!python -m explainable_kd.cli train-baselines \
  --config configs/base.yaml \
  --artifact-root {ARTIFACT_ROOT}
```

```python
from pathlib import Path

for experiment_id in ("student_d10_baseline", "student_d8_baseline", "student_d6_baseline"):
    root = Path(ARTIFACT_ROOT) / "checkpoints" / experiment_id / "seed_42"
    metrics = Path(ARTIFACT_ROOT) / "metrics" / experiment_id / "seed_42/metrics.json"
    print(experiment_id, "best", (root / "best").exists(), "last", (root / "last").exists(), "metrics", metrics.exists())
```

- baseline 학습 결과 전체를 commit한다.

```python
copy_lightweight_results(SMOKE_ARTIFACT_ROOT)
copy_lightweight_results(ARTIFACT_ROOT)
!git add configs src notebooks results
!git commit -m "DOCS: Add Baseline Training Results"
```

### 4번 완료 후

- 최신 코드와 설정을 먼저 받는다.

```python
%cd /content/explainable-knowledge-distillation-bert
!git pull --ff-only
```

```python
!python -m explainable_kd.cli train-kd \
  --config configs/base.yaml \
  --overlay configs/smoke.yaml \
  --artifact-root {SMOKE_ARTIFACT_ROOT}
```

- 동일 artifact root의
  `checkpoints/teacher_d12_supervised/seed_42/best/`를 세 Student가 모두
  읽고 Teacher가 freeze됐는지 확인한다.
- `student_d{10,8,6}_kd`별 `train_task_loss`, `train_kd_loss`,
  `validation_task_loss`, `validation_kd_loss`와 동일한
  `teacher_ref`가 기록됐는지 확인한다.

```python
from pathlib import Path

teacher = Path(SMOKE_ARTIFACT_ROOT) / "checkpoints/teacher_d12_supervised/seed_42/best"
assert teacher.exists(), f"MISSING {teacher}"
for experiment_id in ("student_d10_kd", "student_d8_kd", "student_d6_kd"):
    metrics_path = Path(SMOKE_ARTIFACT_ROOT) / "metrics" / experiment_id / "seed_42/metrics.json"
    print("OK" if metrics_path.exists() else "MISSING", metrics_path)
    if metrics_path.exists():
        import json
        data = json.loads(metrics_path.read_text())
        print(experiment_id, "teacher_ref", data.get("teacher_ref"), "history", bool(data.get("history")))
```
- smoke 성공 후 overlay를 제거하고 `{ARTIFACT_ROOT}`에서 본 학습을 실행한다.

```python
!python -m explainable_kd.cli train-kd \
  --config configs/base.yaml \
  --artifact-root {ARTIFACT_ROOT}
```

```python
import json
from pathlib import Path

for experiment_id in ("student_d10_kd", "student_d8_kd", "student_d6_kd"):
    root = Path(ARTIFACT_ROOT) / "checkpoints" / experiment_id / "seed_42"
    metrics = Path(ARTIFACT_ROOT) / "metrics" / experiment_id / "seed_42/metrics.json"
    print(experiment_id, "best", (root / "best").exists(), "last", (root / "last").exists(), "metrics", metrics.exists())
    if metrics.exists():
        print("history keys:", sorted(json.loads(metrics.read_text()).get("history", [{}])[0].keys()))
```

```python
import json
from pathlib import Path
teacher = Path(SMOKE_ARTIFACT_ROOT) / "checkpoints/teacher_d12_supervised/seed_42/best"
print("teacher checkpoint:", "OK" if teacher.exists() else "MISSING")
for experiment_id in ("student_d10_kd", "student_d8_kd", "student_d6_kd"):
    path = Path(SMOKE_ARTIFACT_ROOT) / "metrics" / experiment_id / "seed_42/metrics.json"
    print(experiment_id, "OK" if path.exists() else "MISSING")
    if path.exists():
        data = json.loads(path.read_text())
        print("history keys:", sorted(data.get("history", [{}])[0].keys()))
```

- 일반 KD 학습 결과 전체를 commit한다.

```python
copy_lightweight_results(SMOKE_ARTIFACT_ROOT)
copy_lightweight_results(ARTIFACT_ROOT)
!git add configs src notebooks results
!git commit -m "DOCS: Add Standard KD Results"
```

### 5번 완료 후

- 최신 코드와 설정을 먼저 받는다.

```python
%cd /content/explainable-knowledge-distillation-bert
!git pull --ff-only
```

- Teacher·baseline·일반 KD의 smoke checkpoint가 모두 있는지 확인한 뒤 IG
  attribution smoke를 실행한다.

```python
!python -m explainable_kd.cli extract-ig \
  --config configs/base.yaml \
  --overlay configs/smoke.yaml \
  --artifact-root {SMOKE_ARTIFACT_ROOT} \
  --max-examples 1
```
- `{SMOKE_ARTIFACT_ROOT}/attributions/ig/<experiment_id>/seed_42/examples.jsonl`과
  `{SMOKE_ARTIFACT_ROOT}/figures/token_importance/` 아래에 Teacher와 baseline/KD
  Student의 동일 질문 결과가 생성되는지 확인한다.

```python
from pathlib import Path
root = Path(SMOKE_ARTIFACT_ROOT)
print("IG attribution files:", len(list((root / "attributions/ig").glob("**/*"))))
print("token-importance figures:", len(list((root / "figures/token_importance").glob("**/*"))))
```
- 저장된 row에서 Teacher와 Student의 `target_class`가 같고, special token과
  padding의 `valid_mask`가 `false`이며 IG 유사도가 기록됐는지 확인한다.

```python
import json
from pathlib import Path

for path in (Path(SMOKE_ARTIFACT_ROOT) / "attributions/ig").glob("**/examples.jsonl"):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print(path, "rows", len(rows))
    for row in rows:
        assert {"tokens", "scores", "target_class", "predicted_class", "valid_mask"} <= row.keys()
        assert "valid_mask" in row
similarity = Path(SMOKE_ARTIFACT_ROOT) / "metrics/ig_similarity/seed_42.json"
assert similarity.exists(), f"MISSING {similarity}"
print("similarity:", similarity)
```
- 이 단계에서는 설명 손실 학습을 실행하지 않는다.

- IG attribution 결과와 시각화 전체를 commit한다.

```python
copy_lightweight_results(SMOKE_ARTIFACT_ROOT)
copy_lightweight_results(ARTIFACT_ROOT)
!git add configs src notebooks results
!git commit -m "DOCS: Add IG Attribution Results"
```

### 6번 완료 후

- 최신 코드와 설정을 먼저 받는다.

```python
%cd /content/explainable-knowledge-distillation-bert
!git pull --ff-only
```

- 먼저 tiny batch의 IG gradient gate를 실행한다.

```python
!python -m explainable_kd.cli verify-xai-gradients \
  --method ig \
  --config configs/base.yaml \
  --overlay configs/smoke.yaml \
  --artifact-root {SMOKE_ARTIFACT_ROOT}
!python -m explainable_kd.cli train-ig-kd \
  --config configs/base.yaml \
  --overlay configs/smoke.yaml \
  --artifact-root {SMOKE_ARTIFACT_ROOT}
```

- gate artifact에서 Student gradient가 finite이고 0이 아니며 Teacher gradient가
  생성되지 않았는지 확인한다.
- `student_d{10,8,6}_ig_kd`별 `task_loss`, `kd_loss`, `ig_loss`, `total_loss`와
  checkpoint를 확인한 뒤 overlay를 제거하여 `{ARTIFACT_ROOT}`에서 본 학습한다.

```python
!python -m explainable_kd.cli train-ig-kd \
  --config configs/base.yaml \
  --artifact-root {ARTIFACT_ROOT}
```

```python
from pathlib import Path

for experiment_id in ("student_d10_ig_kd", "student_d8_ig_kd", "student_d6_ig_kd"):
    root = Path(ARTIFACT_ROOT) / "checkpoints" / experiment_id / "seed_42"
    metrics = Path(ARTIFACT_ROOT) / "metrics" / experiment_id / "seed_42/metrics.json"
    print(experiment_id, "best", (root / "best").exists(), "last", (root / "last").exists(), "metrics", metrics.exists())
```

```python
import json
from pathlib import Path
for path in Path(SMOKE_ARTIFACT_ROOT).glob("metrics/student_d*_ig_kd/seed_42/metrics.json"):
    data = json.loads(path.read_text())
    print(path, sorted(data.get("history", [{}])[0].keys()))
```

```python
import json
from pathlib import Path

gate_files = list(Path(SMOKE_ARTIFACT_ROOT).glob("**/*gradient*.json"))
assert gate_files, "MISSING gradient gate artifact"
for path in gate_files:
    data = json.loads(path.read_text())
    print(path, data)
```

- IG KD 학습 결과 전체를 commit한다.

```python
copy_lightweight_results(SMOKE_ARTIFACT_ROOT)
copy_lightweight_results(ARTIFACT_ROOT)
!git add configs src notebooks results
!git commit -m "DOCS: Add IG KD Results"
```

### 7번 완료 후

- 최신 코드와 설정을 먼저 받는다.

```python
%cd /content/explainable-knowledge-distillation-bert
!git pull --ff-only
```

- 먼저 Teacher·baseline·일반 KD smoke checkpoint가 있는지 확인하고 tiny
  batch의 LRP attribution을 실행한다.

```python
!python -m explainable_kd.cli extract-lrp \
  --config configs/base.yaml \
  --overlay configs/smoke.yaml \
  --artifact-root {SMOKE_ARTIFACT_ROOT} \
  --max-examples 1
```

- LRP attribution smoke가 성공하면 LRP KD smoke를 실행한다.

```python
!python -m explainable_kd.cli train-lrp-kd \
  --config configs/base.yaml \
  --overlay configs/smoke.yaml \
  --artifact-root {SMOKE_ARTIFACT_ROOT}
```

- `attributions/lrp/`에만 LRP 결과가 저장되고 `attributions/ig/`를 읽거나
  덮어쓰지 않는지 확인한다.
- `student_d{10,8,6}_lrp_kd`의 gradient·계산량을 확인한 뒤 overlay를 제거하여
  `{ARTIFACT_ROOT}`에서 본 학습한다.
- 문제가 있으면 임의로 넘어가지 말고 오류와 대안을 기록한다.

```python
!python -m explainable_kd.cli train-lrp-kd \
  --config configs/base.yaml \
  --artifact-root {ARTIFACT_ROOT}
```

```python
from pathlib import Path

for experiment_id in ("student_d10_lrp_kd", "student_d8_lrp_kd", "student_d6_lrp_kd"):
    root = Path(ARTIFACT_ROOT) / "checkpoints" / experiment_id / "seed_42"
    metrics = Path(ARTIFACT_ROOT) / "metrics" / experiment_id / "seed_42/metrics.json"
    print(experiment_id, "best", (root / "best").exists(), "last", (root / "last").exists(), "metrics", metrics.exists())
```

```python
from pathlib import Path

lrp = Path(SMOKE_ARTIFACT_ROOT) / "attributions/lrp"
ig = Path(SMOKE_ARTIFACT_ROOT) / "attributions/ig"
print("LRP files:", len(list(lrp.glob("**/*"))))
print("IG files:", len(list(ig.glob("**/*"))))
assert lrp.exists(), "MISSING LRP artifacts"
```

- LRP attribution 및 LRP KD 결과 전체를 commit한다.

```python
copy_lightweight_results(SMOKE_ARTIFACT_ROOT)
copy_lightweight_results(ARTIFACT_ROOT)
!git add configs src notebooks results
!git commit -m "DOCS: Add LRP KD Results"
```

### 8번 완료 후

- 최신 코드와 설정을 먼저 받는다.

```python
%cd /content/explainable-knowledge-distillation-bert
!git pull --ff-only
```

```python
!python -m explainable_kd.cli evaluate \
  --all \
  --config configs/base.yaml \
  --artifact-root {ARTIFACT_ROOT}
!python -m explainable_kd.cli summarize \
  --config configs/base.yaml \
  --artifact-root {ARTIFACT_ROOT}
!python -m explainable_kd.cli visualize \
  --config configs/base.yaml \
  --artifact-root {ARTIFACT_ROOT}
```

- `{ARTIFACT_ROOT}/reports/experiment_summary.{csv,json}`와
  `case_comparison.csv`, `{ARTIFACT_ROOT}/figures/` 아래의 성능·효율·설명
  유사도·토큰 중요도 그림을 확인한다.
- summary의 실험 ID가 `configs/experiments.yaml`의 13개와 정확히 일치하며,
  결과가 없는 조건은 추정값 대신 `not run`인지 확인한다.

```python
from pathlib import Path
import json

root = Path(ARTIFACT_ROOT)
for path in (
    root / "reports/experiment_summary.csv",
    root / "reports/experiment_summary.json",
    root / "reports/case_comparison.csv",
):
    print("OK" if path.exists() else "MISSING", path)
print("figure_count:", len(list((root / "figures").glob("**/*"))))

summary = root / "reports/experiment_summary.json"
if summary.exists():
    data = json.loads(summary.read_text())
    rows = data if isinstance(data, list) else data.get("experiments", data.get("results", []))
    print("summary_rows:", len(rows))
    print("not_run_rows:", sum(row.get("status") == "not run" for row in rows))
```
- `ARTIFACT_ROOT`를 Google Drive로 지정하지 않았다면 `reports/`, `figures/`,
  필요한 checkpoint를 Drive로 복사한 뒤 세션을 종료한다.

- 통합 평가 결과, 표, 그래프, 시각화 파일 전체를 commit한다.

```python
copy_lightweight_results(SMOKE_ARTIFACT_ROOT)
copy_lightweight_results(ARTIFACT_ROOT)
!git add configs src notebooks results
!git commit -m "DOCS: Add Integrated Experiment Reports"
```
