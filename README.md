# Infrentia2 LLM Sample Code

이 GitHub 저장소는 AWS Inferentia2를 사용하여 대규모 언어 모델(LLM)을 서빙하고 벤치마킹하는 과정을 담은 샘플 코드를 제공합니다. 본 저장소의 코드는 Inferentia2 장치에서 LLM을 최적화하고 효율적으로 실행할 수 있는 방법을 보여줍니다.

## 파일 구성

저장소에는 다음 파일들이 포함되어 있습니다:

- **2.1-Compile-TransformerNeuronX.ipynb**: Transformer 모델을 NeuronX 백엔드를 사용하여 컴파일하는 과정을 설명하는 Jupyter 노트북입니다.

- **3.1-Serving-TorchServe.ipynb.ipynb**: TorchServe를 사용하여 모델을 서빙하는 방법을 단계별로 설명하는 Jupyter 노트북입니다.

- **3.2-Serving-DJLServing.ipynb**: DJLServing을 사용하여 LLM을 서빙하는 방법을 단계별로 설명하는 Jupyter 노트북입니다. Inferentia2 환경에서 모델을 로드하고 HTTP 엔드포인트를 통해 서빙하는 과정을 포함합니다.

- **4-Benchmark.ipynb**: 모델의 성능을 평가하기 위한 벤치마킹을 수행하는 Jupyter 노트북입니다. 이 노트북은 서빙된 모델의 처리량과 지연 시간을 측정하여 모델의 성능을 분석합니다.

## 시작하기

### 필요 조건

- AWS 계정

### 모델 컴파일하기

1. `2.1-Compile-TransformerNeuronX.ipynb` 노트북을 실행하여 Transformer 모델을 NeuronX 백엔드를 사용해 컴파일합니다. 

### 모델 서빙하기

1. `3.1-Serving-TorchServe.ipynb` 또는 `3.2-Serving-DJLServing.ipynb` 노트북을 실행하여 모델을 서빙합니다. 노트북은 필요한 모든 단계를 설명하며, AWS Inferentia2 인스턴스에 모델을 배포하는 방법을 보여줍니다.

### 벤치마킹 수행

1. `4-Benchmark.ipynb`를 실행하여 모델의 성능을 평가합니다. 이 노트북은 모델 서버에 대한 요청을 보내고, 응답 시간과 처리량 등의 성능 지표를 측정합니다.

2. 추가적으로 `benchmark.py` 스크립트를 사용하여 명령줄에서 벤치마킹을 수행할 수 있습니다.

## 지원

이 프로젝트에 대한 질문이나 지원이 필요한 경우, GitHub 이슈 트래커를 통해 문의주세요.