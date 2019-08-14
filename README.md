# Counterfeit-Bill-Discrimination
 
 
## 서론

**계기**
- 2017년 1월 11일 경북대 SW 동아리 전시회에 참가

**주제**
- 딥러닝을 활용한 위조 지폐 판별

**설명**
- 특징 기반 위조 지폐 판별은 UV선, 전체 지폐 사진 등을 이용하기 때문에 특수한 도구나 고가의 장비가 필요
- 딥러닝을 이용하여 지폐의 일부만을 이용해서 위폐 판별이 가능한 기술 필요
- 고가의 스캐너가 아닌 범용적인 성능의 스캐너로도 충분히 사용 가능

## 실험 환경

**하드웨어 스팩**
- 그래픽 카드 : NVIDIA Titan XP
- RAM : 16GB

**소프트웨어 버전**
- OS : Windows 10 64bit
- CUDA / cuDNN : 8.0 / 5.1
- Python / Tensorflow-gpu : 3.5 / 1.4.0

## 실험 내용

**모델 구조**

<img src="https://github.com/SoohyeonLee/Counterfeit-Bill-Discrimination/blob/master/resource/model structure.png" width="90%"></img>

**사용 데이터**

<img src="https://github.com/SoohyeonLee/Counterfeit-Bill-Discrimination/blob/master/resource/bill data split.png" width="90%"></img>

## 실험 결과

<img src="https://github.com/SoohyeonLee/Counterfeit-Bill-Discrimination/blob/master/resource/accuracy.png" width="90%"></img>

___

**작성일 : 2018.08.14**
