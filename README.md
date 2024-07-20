# AWS Inferentia2로 대규모 언어 모델(LLM) 서빙 최적화하기

AWS Inferentia2를 활용해 LLM을 효율적으로 서빙하고 벤치마킹하는 방법을 소개합니다. 🚀

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 주요 기능 🌟

- Transformer NeuronX 컴파일
- DJLServing을 통한 모델 서빙
- 성능 벤치마킹 및 분석

## 시작하기 🏁

### 필요 조건

- AWS 계정
- Inferentia2를 지원하는 AWS 리전 (예: 도쿄 ap-northeast-1, 버지니아 us-east-1)
- SSH 클라이언트

### 설정 및 설치

1. **인스턴스 생성**
   - EC2 콘솔에서 "Deep Learning AMI Neuron (Ubuntu 22.04)" AMI 선택
   - `inf2.24xlarge` 인스턴스 유형 선택 (또는 원하는 Neuron 인스턴스)
   - 디스크 크기 300GB로 설정
   - Public Subnet에 배치 및 SSH Key-Pair 활성화

2. **SSH 접속**
   ```bash
   ssh -i ~/.ssh/your_key.pem -L 8888:127.0.0.1:8888 ubuntu@your-instance-ip
   ```

3. **환경 설정**
   ```bash
   source /opt/aws_neuronx_venv_transformers_neuronx/bin/activate
   pip install sentencepiece
   pip install --upgrade-strategy eager optimum[neuronx]
   ```

4. **코드 다운로드**
   ```bash
   git clone https://github.com/didhd/inferentia2-llm.git
   ```

5. **Jupyter Lab 실행**
   ```bash
   jupyter lab
   ```

6. 브라우저에서 Jupyter Lab 접속 (표시된 URL 사용)

7. Hugging Face 로그인 및 Llama3 EULA 동의 (필요시)

## 사용 방법 📘

1. **모델 컴파일**: `2.1-Compile-TransformerNeuronX.ipynb` 실행
2. **모델 서빙**: 
   - TorchServe: `3.1-Serving-TorchServe.ipynb` 실행
   - DJLServing: `3.2-Serving-DJLServing.ipynb` 실행
3. **벤치마킹**: `4-Benchmark.ipynb` 실행

각 노트북에 상세한 실행 가이드가 포함되어 있습니다.

## 라이선스 📄

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 문의하기 📮

질문이나 피드백이 있으시면 이슈를 열어주세요.
