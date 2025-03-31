## Installation
```bash
conda env create -f environment.yml
```

## 데이터 전처리

`rawdata/` 디렉토리에 있는 원시 데이터를 가공하여 `data/train/` 및 `data/test/` 디렉토리에 학습에 적합한 형태로 저장

### 전체 실행
```bash
python Generating/data_generator.py
```

### 함수별 설명

- merge_and_save_monthly_data()
  - 6개월치 raw .parquet 파일을 불러오고 불필요한 상수열(constant feature)을 제거
  - 각 카테고리별로 월별 데이터를 합쳐서 train/test 데이터로 저장

- merge_segment_feature()
  - 개별 모델링 시 Segment 정보를 활용할 수 있도록 customer 데이터에만 존재하는 Segment 컬럼을 다른 카테고리의 train 데이터에 병합

- convert_dtypes()
  - 설정 파일(cfg.DTYPES)에 정의된 데이터 타입으로 컬럼을 변환
    - 날짜 형식(YM, YMD)은 datetime으로 변환
    - 기타 타입은 메모리 사용을 줄이는 형태로 캐스팅
  - 변환 전/후 메모리 사용량도 함께 출력

- merge_categories()
  - 모든 refined train/test 데이터를 하나로 통합
  - 중복되는 컬럼(ID, 기준년월, Segment)은 적절히 제거하여 병합
