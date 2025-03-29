import os
import pandas as pd
from itertools import product
import pyarrow.parquet as pq
from xgboost import train
from config import Configuration as cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def calculate_memory_MB(df):
    return df.memory_usage().sum() / 1024 / 1024


def convert_dtypes():

    for df_name, schema in cfg.DTYPES.items():

        print(f"Converting {df_name} data types...")

        df_train = pd.read_parquet(f"{ROOT_DIR}/data/train/{df_name}_train.parquet")
        df_test = pd.read_parquet(f"{ROOT_DIR}/data/test/{df_name}_test.parquet")
        before_train = calculate_memory_MB(df_train)
        before_test = calculate_memory_MB(df_test)

        for col in df_train.columns:
            if col == "기준년월":
                continue

            dtype = schema[col][1]

            if dtype == "YM":
                df_train[col] = pd.to_datetime(
                    df_train[col].astype("Int64"), format="%Y%m", errors="coerce"
                )
                df_test[col] = pd.to_datetime(
                    df_test[col].astype("Int64"), format="%Y%m", errors="coerce"
                )

            elif dtype == "YMD":
                df_train[col] = pd.to_datetime(
                    df_train[col].astype("Int64"), format="%Y%m%d", errors="coerce"
                )
                df_test[col] = pd.to_datetime(
                    df_test[col].astype("Int64"), format="%Y%m%d", errors="coerce"
                )
            else:
                df_train[col] = df_train[col].astype(dtype)

                if col != "Segment":
                    df_test[col] = df_test[col].astype(dtype)

        after_train = calculate_memory_MB(df_train)
        after_test = calculate_memory_MB(df_test)

        print(
            f"\t{df_name}_train before: {before_train:.2f} MB, after: {after_train:.2f} MB"
        )
        print(
            f"\t{df_name}_test before: {before_test:.2f} MB, after: {after_test:.2f} MB"
        )

        df_train.to_parquet(f"{ROOT_DIR}/data/train/refined_{df_name}_train.parquet")
        df_test.to_parquet(f"{ROOT_DIR}/data/test/refined_{df_name}_test.parquet")


# 모든 월의 데이터를 다 사용하고 ID만 지우면 됨
def merge_and_save_monthly_data():

    loaded_train_data = {}
    loaded_test_data = {}

    for split, category, month in product(cfg.SPLITS, cfg.DATA_CATEGORIES, cfg.MONTHS):

        folder = cfg.DATA_CATEGORIES[category]["folder"]
        var_prefix = cfg.DATA_CATEGORIES[category]["var_prefix"]
        file_path = f"{ROOT_DIR}/rawdata/{split}/{folder}/2018{month}_{split}_{cfg.DATA_CATEGORIES[category]['name']}.parquet"

        # constant한 열은 제외
        all_columns = pq.read_schema(file_path).names
        excluded_columns = cfg.CONSTANT_FEATURES.get(var_prefix, [])
        selected_columns = [col for col in all_columns if col not in excluded_columns]

        if split == "train":
            loaded_train_data[f"{var_prefix}_{split}_{month}"] = pd.read_parquet(
                file_path, columns=selected_columns
            )
        else:
            loaded_test_data[f"{var_prefix}_{split}_{month}"] = pd.read_parquet(
                file_path, columns=selected_columns
            )

    print(f"Data loaded successfully from rawdata of 6 months")

    os.makedirs(f"{ROOT_DIR}/data/train", exist_ok=True)
    os.makedirs(f"{ROOT_DIR}/data/test", exist_ok=True)

    # 각 category 별로 6개월치 데이터를 하나로 합쳐서 저장
    for category in cfg.DATA_CATEGORIES:
        var_prefix = cfg.DATA_CATEGORIES[category]["var_prefix"]
        train_df_list = [
            loaded_train_data[f"{var_prefix}_train_{month}"] for month in cfg.MONTHS
        ]
        test_df_list = [
            loaded_test_data[f"{var_prefix}_test_{month}"] for month in cfg.MONTHS
        ]
        pd.concat(train_df_list).to_parquet(
            f"{ROOT_DIR}/data/train/{var_prefix}_train.parquet"
        )
        pd.concat(test_df_list).to_parquet(
            f"{ROOT_DIR}/data/test/{var_prefix}_test.parquet"
        )

    print("Data saved successfully to data/train, data/test directories")


def merge_segment_feature():
    # customer 에만 있는 Segment를 다른 데이터셋에 merge -> 각 category 별 모델만들 때 활용
    customer_train = pd.read_parquet(f"{ROOT_DIR}/data/train/customer_train.parquet")
    subset = customer_train[["ID", "기준년월", "Segment"]]

    for category in cfg.CATEGORIES:
        if category == "customer":
            customer_train.sort_values(
                by=["ID", "기준년월"], inplace=True, ignore_index=True
            )
            customer_train.to_parquet(f"{ROOT_DIR}/data/train/customer_train.parquet")
            print(customer_train[["ID", "기준년월"]].head())

        else:
            train_df = pd.read_parquet(
                f"{ROOT_DIR}/data/train/{category}_train.parquet"
            )
            train_df = train_df.merge(subset, on=["ID", "기준년월"]).sort_values(
                by=["ID", "기준년월"], ignore_index=True
            )
            train_df.to_parquet(f"{ROOT_DIR}/data/train/{category}_train.parquet")

            print(train_df[["ID", "기준년월"]].head())

            # print(f"Successfully merged Segments into {category} dataset")
            # print(f"Shape of {category} dataset: {train_df.shape}")
            # print()


def merge_categories():
    # 1. Load and sort train
    # 각 df에는 2,400,000 rows가 들어있음
    train_dfs = [
        pd.read_parquet(f"{ROOT_DIR}/data/train/refined_{cat}_train.parquet")
        for cat in cfg.CATEGORIES
    ]
    # ID, 기준년월 제거 (모든 df에서)
    for i in range(len(train_dfs)):
        train_dfs[i] = train_dfs[i].drop(columns=["ID", "기준년월"])

    # Segment는 첫 번째 df에서만 유지되어야 하므로, 0번째는 유지
    for i in range(1, len(train_dfs)):
        train_dfs[i] = train_dfs[i].drop(columns=["Segment"])

    # # 2. Train concat and remove duplicates
    train_concat = pd.concat(train_dfs, axis=1)
    train_concat = train_concat.drop_duplicates().reset_index(drop=True)
    print(calculate_memory_MB(train_concat))
    train_concat.to_parquet(f"{ROOT_DIR}/data/train/merge_train.parquet")

    # memory remove
    del train_dfs
    del train_concat

    # # 3. Load and sort test
    test_dfs = [
        pd.read_parquet(f"{ROOT_DIR}/data/test/refined_{cat}_test.parquet")
        for cat in cfg.CATEGORIES
    ]

    for i in range(len(test_dfs)):
        test_dfs[i] = test_dfs[i].drop(columns=["기준년월"])

    for i in range(1, len(test_dfs)):
        test_dfs[i] = test_dfs[i].drop(columns=["ID"])

    # 2. Test concat and remove duplicates
    test_concat = pd.concat(test_dfs, axis=1)
    test_concat = test_concat.drop_duplicates().reset_index(drop=True)
    print(calculate_memory_MB(test_concat))
    test_concat.to_parquet(f"{ROOT_DIR}/data/test/merge_test.parquet")


if __name__ == "__main__":
    # # # 6개월치 모든 데이터 사용 & constant feature 제외
    # merge_and_save_monthly_data()

    # # # Segment feature 결합
    # merge_segment_feature()

    # # # train, test 데이터에서 dtype 변환
    # convert_dtypes()

    # ID 기준, train, test 데이터 각각 결합
    merge_categories()
    # train, test = merge_categories()
    # print(train.info())
    # print(test.info())
