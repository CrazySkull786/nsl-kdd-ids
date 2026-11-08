# export_model.py - saves the "champion" model from MLflow as champion.pkl + its fingerprint
import hashlib
import joblib
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/ids-model@champion")
joblib.dump(model, "champion.pkl")

sha = hashlib.sha256(open("champion.pkl", "rb").read()).hexdigest()
open("champion.pkl.sha256", "w").write(sha)
print("Saved champion.pkl, sha256:", sha[:16], "...")
