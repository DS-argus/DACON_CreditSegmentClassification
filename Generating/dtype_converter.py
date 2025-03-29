import os
import pandas as pd
from Generating.config import DtypesSchema

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DTYPES = {
    "customer": DtypesSchema.CUSTOMER_SCHEMA,
    "credit": DtypesSchema.CREDIT_SCHEMA,
    "sales": DtypesSchema.SALES_SCHEMA,
    "billing": DtypesSchema.BILLING_SCHEMA,
    "marketing": DtypesSchema.MARKETING_SCHEMA,
    "performance": DtypesSchema.PERFORMANCE_SCHEMA,
    "channel": DtypesSchema.CHANNEL_SCHEMA,
    "balance": DtypesSchema.BALANCE_SCHEMA,
}


def calculate_memory_MB(df):
    return df.memory_usage().sum() / 1024 / 1024


def convert_dtypes():

    for df_name, schema in DTYPES.items():

        print(f"Converting {df_name} data types...")

        df_train = pd.read_parquet(f"{ROOT_DIR}/data/train/{df_name}_train.parquet")
        df_test = pd.read_parquet(f"{ROOT_DIR}/data/test/{df_name}_test.parquet")
        before_train = calculate_memory_MB(df_train)
        before_test = calculate_memory_MB(df_test)

        for col in df_train.columns:
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


if __name__ == "__main__":
    convert_dtypes()
