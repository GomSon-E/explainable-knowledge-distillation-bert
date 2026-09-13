# 내용 정리

## 큰 방향

- 과제는 BERT 또는 GPT-2를 하나 이상 활용해 자유롭게 태스크를 정의하는 것이지만, 이번 프로젝트는 **BERT 기반 질문 유형 분류 모델 경량화**를 주제로 한다.
- 큰 **Teacher BERT**의 지식과 판단 근거를 작은 **Student BERT**로 전달하는 **Knowledge Distillation (KD)** 을 구현한다.
- 단순히 Teacher의 최종 예측 확률만 모방하는 데서 끝내지 않고, Teacher가 문장 내 어떤 단어를 중요하게 봤는지까지 Student가 따라 배우게 한다.

## 데이터셋 및 실행 환경 메모

- 데이터셋: **TREC-6**.
- TREC-6를 사용하면 태스크는 감성 분류가 아니라, 질문을 6개 질문 유형으로 분류하는 **질문 분류(question classification)** 가 된다.
  - 예: abbreviation, entity, description, human, location, numeric.
- 입력 길이는 BERT 최대 512 tokens로 제한하며, IG·LRP 중요도 계산과 설명 손실 학습의 계산량을 관리한다.
- 실제 학습과 실험은 **Google Colab GPU 자원**을 사용한다.

## Teacher–Student KD

- Teacher: 태스크를 충분히 학습한 12층 BERT.
- Student: Teacher보다 Transformer 층 수를 줄인 BERT. 10층, 8층, 6층을 각각 실험한다.
- 모든 Student의 Teacher는 **동일한 12층 Teacher 1개**이다. 12층 → 10층 → 8층 → 6층의 순차 증류 구조는 아니다.
- KD에서는 Teacher의 출력 확률 분포(soft target)와 Student의 출력 확률 분포가 비슷해지도록 학습한다.

## XAI를 활용한 설명 전달

- 멘토가 언급한 설명 기법은 **IG(Integrated Gradients)** 와 **LRP(Layer-wise Relevance Propagation)** 이다.
- 둘 다 “이 예측에 각 입력 단어가 얼마나 기여했는가”를 수치화하는 XAI 기법이다.
- Teacher와 Student의 단어 중요도 분포가 가까워지도록 설명 손실을 추가한다.
- IG와 LRP는 단순한 사후 시각화 도구가 아니라, 각각 **별도의 설명 손실**로 사용해 실험한다.
- 설명 계산 결과는 학습에도 쓰고, 최종적으로 색칠된 단어 중요도 그림으로도 제시한다.

## 손실 함수 개념

Student의 학습 손실은 다음 세 항을 조합한다.

\[
L = \alpha L_{task} + \beta L_{KD} + \gamma L_{explanation}
\]

- `L_task`: 실제 질문 유형 라벨을 맞히기 위한 분류 손실.
- `L_KD`: Teacher와 Student의 예측 분포를 맞추는 지식 증류 손실.
- `L_explanation`: Teacher와 Student의 단어 중요도를 맞추는 설명 모방 손실. IG 또는 LRP를 사용한다.
