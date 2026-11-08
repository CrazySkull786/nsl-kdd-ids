# Data tests: is the data safe to train on?
import pandas as pd
from schema import COLS

def load(path):
    first = open(path).readline().split(",")[0].strip().lower()
    return pd.read_csv(path) if first == "duration" else pd.read_csv(path, names=COLS)

train, test = load("data/train.csv"), load("data/test.csv")

def test_columns_are_correct():
    assert list(train.columns) == COLS and list(test.columns) == COLS

def test_no_empty_cells():
    assert train.notna().all().all() and test.notna().all().all()

def test_no_negative_numbers():
    assert (train.select_dtypes("number") >= 0).all().all()

def test_protocols_are_known():
    assert set(train["protocol_type"]) <= {"tcp", "udp", "icmp"}

def test_enough_rows_and_both_classes():
    assert len(train) >= 100000
    assert "normal" in set(train["label"]) and train["label"].nunique() > 5
