# AWS Inferentia2로 대규모 언어 모델(LLM) 서빙 최적화하기
AWS Inferentia2를 활용해 LLM을 효율적으로 서빙하고 벤치마킹하는 방법을 소개합니다. 🚀
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 주요 기능 🌟
- DJLServing을 통한 모델 서빙
- 성능 벤치마킹 및 분석
- Transformer NeuronX 컴파일 (선택적)
- TorchServe를 통한 모델 서빙 (선택적)

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

2. **IAM 역할 설정**
   - '고급 세부 정보'에서 '새 IAM 인스턴스 프로파일 생성' 클릭
   - 신뢰할 수 있는 엔터티 유형으로 'AWS Service'의 'EC2' 선택
   - 인라인 정책 'ECRPolicy' 생성 (ECR 관련 권한 부여)
   - 역할 이름을 'ServingInstanceRole'로 설정
   - EC2 생성 화면에서 새로 만든 IAM 역할 선택

3. **SSH 접속**
   ```bash
   ssh -i ~/.ssh/your_key.pem -L 8888:127.0.0.1:8888 ubuntu@your-instance-ip
   ```

4. **환경 설정**
   ```bash
   source /opt/aws_neuronx_venv_transformers_neuronx/bin/activate
   pip install sentencepiece
   pip install --upgrade-strategy eager optimum[neuronx]
   ```

5. **코드 다운로드**
   ```bash
   git clone https://github.com/didhd/inferentia2-llm.git
   ```

6. **Jupyter Lab 실행**
   ```bash
   jupyter lab
   ```

7. 브라우저에서 Jupyter Lab 접속 (표시된 URL 사용)

8. Hugging Face 로그인 및 Llama2 EULA 동의 (필요시)

## 사용 방법 📘
1. **모델 서빙**: `1-Serving-DJLServing.ipynb` 실행
2. **벤치마킹**: `2-Benchmark.ipynb` 실행
3. **모델 컴파일** (선택 실습): `3.1-Compile-TransformerNeuronX.ipynb` 실행
4. **TorchServe 서빙** (선택 실습): `3.2-Serving-TorchServe.ipynb` 실행

각 노트북에 상세한 실행 가이드가 포함되어 있습니다.

## 라이선스 📄
이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 문의하기 📮
질문이나 피드백이 있으시면 이슈를 열어주세요.
