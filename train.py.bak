# train.py - Usage: python train.py [logistic|forest|xgboost]
import sys, hashlib
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("intrusion-detection")
model_type = sys.argv[1] if len(sys.argv) > 1 else "logistic"
SEED = 42

COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
 "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
 "root_shell","su_attempted","num_root","num_file_creations","num_shells",
 "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count",
 "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
 "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
 "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
 "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
 "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
 "label","difficulty"]

def load(path):
    first = open(path).readline().split(",")[0].strip().lower()
    df = pd.read_csv(path) if first == "duration" else pd.read_csv(path, names=COLS)
    df["target"] = (df["label"].astype(str).str.lower() != "normal").astype(int)  # 1 = attack
    return df.drop(columns=["label", "difficulty"])   # LEAKAGE GUARD: remove the answer columns

train_df, test_df = load("data/train.csv"), load("data/test.csv")
X, y = train_df.drop(columns="target"), train_df["target"]
X_test, y_test = test_df.drop(columns="target"), test_df["target"]
assert not {"label", "difficulty", "target"} & set(X.columns), "leaky column in features!"

# ---- LEAKAGE CHECKS (warnings) ----
overlap = len(pd.merge(X.drop_duplicates(), X_test.drop_duplicates(), how="inner"))
print("Rows shared by train and test:", overlap)
for c in X.select_dtypes("number").columns:
    auc = roc_auc_score(y, X[c]); auc = max(auc, 1 - auc)
    if auc > 0.99: print(f"WARNING: feature '{c}' alone gives AUC {auc:.3f} - check for leakage")

# Validation split picks the champion; the test file is used ONLY for the final report
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

cat = ["protocol_type", "service", "flag"]
num = [c for c in X.columns if c not in cat]
prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                          ("num", StandardScaler(), num)])

if model_type == "forest":
    clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1, class_weight="balanced")
elif model_type == "xgboost":
    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                        eval_metric="logloss", random_state=SEED, n_jobs=-1)
else:
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
model = Pipeline([("prep", prep), ("clf", clf)])   # preprocessing learns from training rows only

def score(m, Xs, ys):
    pred, proba = m.predict(Xs), m.predict_proba(Xs)[:, 1]
    return {"f1": f1_score(ys, pred), "recall": recall_score(ys, pred),
            "precision": precision_score(ys, pred), "roc_auc": roc_auc_score(ys, proba),
            "accuracy": accuracy_score(ys, pred)}

with mlflow.start_run():
    model.fit(X_tr, y_tr)
    val, test = score(model, X_val, y_val), score(model, X_test, y_test)
    print(f"{model_type} | val F1 {val['f1']:.3f} recall {val['recall']:.3f} | "
          f"test F1 {test['f1']:.3f} recall {test['recall']:.3f}")

    mlflow.log_param("model_type", model_type)
    mlflow.log_param("seed", SEED)
    mlflow.log_param("train_test_overlap_rows", overlap)
    for k, v in val.items():  mlflow.log_metric(f"val_{k}", v)
    for k, v in test.items(): mlflow.log_metric(f"test_{k}", v)

    joblib.dump(model, "model.pkl")
    sha = hashlib.sha256(open("model.pkl", "rb").read()).hexdigest()   # tamper-proof fingerprint
    open("model.pkl.sha256", "w").write(sha)
    mlflow.set_tag("model_sha256", sha)

    try:
        mlflow.sklearn.log_model(model, name="model", registered_model_name="ids-model",
                                 skops_trusted_types=["sklearn.tree._tree.Tree"])
    except Exception as e:
        print("Safe format refused this model, using fallback:", str(e)[:150])
        mlflow.sklearn.log_model(model, name="model", registered_model_name="ids-model",
                                 serialization_format="cloudpickle")
    print("Done. sha256:", sha[:16], "...")