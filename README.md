### 데이터 전처리 
1. 날짜 데이터 처리 
- 일자 데이터에서 월, 요일, 주말 여부 정보 추출
- 요일은 One-Hot Encoding 적용

2. 주말여부 
- 평일 0, 주말 1 이진 변수로 생성

3. 공휴일 여부 
- 한국천문연구원의 특일 정보 API(getRestDeInfo)를 활용하여 공식 공휴일 데이터
- 수집 기간: 2023년 1월 ~ 2026년 3월
- 일반일 0, 공휴일 1


### 주요 파일
- Oworld.csv - 최종피처 파일
- Owrold_hold.csv - 보류 피처 파일
- Owrold_processed_final - Oworld.csv를 바탕으로 데이터 처리 및 정규화(Normalization)를 거쳐 새로 생성된 가공 파일
- test4.ipynb - Owrold.csv 파일을 바탕으로 진행된 데이터 분석 및 전처리 