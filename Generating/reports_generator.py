import os
import time
import pandas as pd
from ydata_profiling import ProfileReport
from Generating.config import Configuration as cnf

import warnings

warnings.filterwarnings(action="ignore")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_report(df: pd.DataFrame, name: str):
    profile = ProfileReport(
        df,
        title=name,
        explorative=True,
        pool_size=0,
        vars={"num": {"low_categorical_threshold": 30}},
        missing_diagrams={"bar": True, "matrix": False, "heatmap": False},
        interactions=None,
    )
    profile.to_file(f"{ROOT_DIR}/reports/{name}_report.html")


def generate_reports(categories: list, refined=False):
    """
    Generate reports for each category in data/train
    """

    os.makedirs(f"{ROOT_DIR}/reports", exist_ok=True)

    for category in categories:
        t1 = time.time()
        df = (
            pd.read_parquet(f"{ROOT_DIR}/data/train/{category}_train.parquet")
            if not refined
            else pd.read_parquet(
                f"{ROOT_DIR}/data/train/refined_{category}_train.parquet"
            )
        )
        print(f"Generating report for {category}")
        generate_report(df, category)
        print(f"Report for {category} is generated in {time.time()-t1:.2f} seconds")
        print()


if __name__ == "__main__":
    categories = cnf.CATEGORIES
    generate_reports(categories)
