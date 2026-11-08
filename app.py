# app.py - secure intrusion-detection API using champion.pkl
import hashlib
import os
import secrets
from typing import Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import ConfigDict, Field, create_model

from schema import FEATURES, RATES, BINARY

app = FastAPI(title="Intrusion Detection API")

# 1. Load the model ONCE, but only if it is exactly the file we exported (tamper check)
expected = open("champion.pkl.sha256").read().strip()
actual = hashlib.sha256(open("champion.pkl", "rb").read()).hexdigest()
if actual != expected:
    raise RuntimeError("champion.pkl does not match its fingerprint - refusing to load")
model = joblib.load("champion.pkl")

# 2. The request form: strict types and ranges, unknown fields are rejected
fields = {}
for name in FEATURES:
    if name == "protocol_type":
        fields[name] = (Literal["tcp", "udp", "icmp"], ...)
    elif name == "service":
        fields[name] = (str, Field(..., pattern=r"^[A-Za-z0-9_]+$", max_length=20))
    elif name == "flag":
        fields[name] = (str, Field(..., pattern=r"^[A-Z0-9]+$", max_length=10))
    elif name in RATES:
        fields[name] = (float, Field(..., ge=0, le=1))
    elif name in BINARY:
        fields[name] = (int, Field(..., ge=0, le=1))
    else:
        fields[name] = (float, Field(..., ge=0, le=1e10))
Flow = create_model("Flow", __config__=ConfigDict(extra="forbid"), **fields)

# 3. API key check (key comes from an environment variable, never from the code)
API_KEY = os.environ.get("API_KEY")

def check_key(x_api_key: str = Header(default="")):
    if not API_KEY or not secrets.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", dependencies=[Depends(check_key)])
def predict(flow: Flow):
    row = pd.DataFrame([flow.model_dump()])
    p = float(model.predict_proba(row)[0, 1])
    return {"prediction": "attack" if p >= 0.5 else "normal", "attack_probability": round(p, 4)}
