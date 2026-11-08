# Quality gate: the new model must pass the "exam" before it can be released
import json

def scores():
    with open("metrics.json") as f:
        return json.load(f)

def test_validation_f1():
    assert scores()["val_f1"] >= 0.95, "validation F1 is below the pass mark 0.95"

def test_test_roc_auc():
    assert scores()["test_roc_auc"] >= 0.90, "test ROC-AUC is below the pass mark 0.90"
