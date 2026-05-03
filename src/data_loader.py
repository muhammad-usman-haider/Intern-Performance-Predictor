import pandas as pd
from .config import CSV_PATH

def load_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df