# 아파트 실거래가 예측 프로젝트

![image](https://github.com/liatamot/House-Price-Prediction-Regression/assets/138054658/eebd2474-776e-4281-9c20-1f6e61ab458c)

## Team

### Role
- 박성우: EDA, Feature engineering, Modeling
- 노균호: 팀장, EDA
- 김윤겸: 모델링, Feature Selection
- 남영진: EDA
- 장호준: EDA, Feature engineering, Modeling
- 이재민: EDA

## 1. Competiton Info

### Overview

- 제공 데이터 활용, 서울의 아파트 실거래가를 예측하는 대회
아파트의 가치, 주변 요소 등에 의해 시간에 따른 가격의 변동 포착이 중요


### 제출물

- 9272개의 아파트 특징 및 거래 정보에 대하여, 거래금액 예측(CSV파일)

### Evaluation

- 평가지표 : RMSE(Root Mean Squared Error)

## 2. Components

### Directory

- _Insert your directory structure_

## 3. Data descrption

### Dataset overview

- 학습 데이터
  - 2007/1/1 ~ 2023/6/30 기간 동안의 데이터
시군구, 아파트명, 전용면적, 건축년도 등의 아파트 가격과 관련된 중요한 변수들 존재 
  - 지하철역 데이터, 버스정류장 데이터


### EDA
- 특정 지역에 최고가 건물이 밀집됨을 확인할 수 있으나 평균치로 보았을 때에는 지역에 따라 건물의 실거래가격이 높다고 판단할 수 없음
- 지역 관련 feature 전체 drop 결정
- ‘시군구’ drop ->  번지,본번,부번또한 시군구 data가 없으면 사용하기 힘들다고 판단하여 drop
- 전용를 제외한 k-연면적, 건축면적,k-관리비부과면적,k-주거전용면적,은 상관관계가 있으나 결측치가 많아 결과를 저해시켜 drop시키는 것으로 결정
- drop
  - 시군구, 번지, 본번, 부번, 계약일, 층, 단지분류, k-전체동수, k-전체세대수, 주차대수,거래유형, 분양형태, k-관리방식, k-복도유형, k-난방방식, k-건설사(시공사), k-시행사, 경비비관리형태, 세대전기계약방법, 청소비관리형태, 사용허가여부, 관리비 업로드, 기타/의무/임대/임의=1/2/3/4, k-전화번호, k-팩스번호, k-홈페이지, 단지신청일,k-등록일자, 단지소개기존clob, 단지승인일, k-사용검사일-사용승인일, k-연면적, k-주거전용면적, 
k-관리비부과면적, k-수정일자, 고용보험관리번호, 건축면적,좌표X,좌표Y  

- 파생 drop
  -  파생에 쓰인 feature를 drop하였기 때문에 한번에 drop
sub_location,bus_location,star_location,세대별 주차대수

- save
  - 아파트명, 도로명, 전용면적, 계약년월, 건축년도

- 여부확인
  - 해제사유발생일,등기신청일자

- 결측치 채우기
  - 이전에는 거래되었지만 더이상 거래되지 않거나 건물이 무너지고 재건축한 경우 재건축한 건물을 기준으로 뽑기 위해 test데이터에 있는  건물들만 학습에 활용하기로 결정
  - f_train = df_train[df_train['아파트명'].isin(df_test['아파트명'])]
  - df_train = df_train[df_train['도로명'].isin(df_test['도로명'])]
  - df_train = df_train[df_train['번지'].isin(df_test['번지'])]
  - df_train = df_train[df_train['시군구'].isin(df_test['시군구'])]
  - 위 코드로 방해되는 것으로 생각되는 데이터 삭제
  - train data - 110만여개 -> 82만여개

- 등기신청여부,해제사유일 삭제
  - df_train = df_train[(df_train['등기신청일자여부'] != 1) & (df_train['해제사유발생여부'] != 1)]


### Feature engineering

`신규 생성 column `
- 내부 데이터 활용 파생 변수
    - df_train['등기신청일자여부'] = df_train['등기신청일자'].apply(lambda x: 1 if pd.notnull(x) else 0)
	개인간 거래인 경우로 추측되는 변수 - 증여 및 시세와 이상치가 있는 개인간 거래로 판단하여 drop의 기준으로 삼기 위해 추가

   - df_train['해제사유발생여부'] = df_train['해제사유발생일'].apply(lambda x: 1 if pd.notnull(x) else 0)
	실제거래가 이루어지지 않은 것으로 실거래가를 유추하기에는 방해되는 값들이라 판단하여 drop의 기준으로 삼기 위해 추가

    - concat['세대별_주차대수'] = concat['주차대수'] / concat['k-전체세대수’]

`외부 데이터 활용 파생 column`
- sub_location - 가장 가까운 지하철과의 거리
- bus_location - 가장 가까운 버스 정류장과의 거리
- star_location - 가장 가까운 스타벅스와의 거리-스타벅스 좌표 크롤링 후 계산

- 아파트 실거래 지표 밑 금리 및 물가 데이터 수집
  - 2020년 이후로는 조사한 모든 데이터의 지표가 아파트 실거래 지표와 달라짐
  - 2020년 이전의 데이터는 오히려 그 이후의 데이터를 해친다고 판단
  - 건축년도 변수 drop 거래년월도 다른 데이터와 merge및 이전 데이터 삭제활용으로만 사용

## 4. Modeling

### Model descrition

- 6개 모델 Voting 앙상블 (with soft, weights)
RandomForestRegressor, XGBRegressor, LGBMRegressor, CatBoostRegressor, HistGradientBoostingRegressor

### Modeling Process

- 모델별 Out-of-fold 예측값을 생성하고 LADRegression으로 학습시켜서 가중치 생성
- Optuna를 통한 파라미터 튜닝을 시도해보았으나 성능향상이 되지 않아 제출파일로 채택하지 않았음


## 5. Result

### Leader Board
- 1등
![image](https://github.com/devguno/Enefit/assets/130540234/07e3efc8-0a35-430e-ac78-a3add1b79990)



### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
