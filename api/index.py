from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal
import logging

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

app = FastAPI(title="MediAI Diagnostics API", version="1.0.0")


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DiabetesRequest(InputModel):
    action: Literal["diabetes"]
    pregnancies: Annotated[int, Field(ge=0, le=20)]
    glucose: Annotated[int, Field(ge=0, le=300)]
    blood_pressure: Annotated[int, Field(ge=0, le=200)]
    skin_thickness: Annotated[int, Field(ge=0, le=100)]
    insulin: Annotated[int, Field(ge=0, le=900)]
    bmi: Annotated[float, Field(ge=0, le=70)]
    dpf: Annotated[float, Field(ge=0, le=3)]
    age: Annotated[int, Field(ge=1, le=120)]


class HeartRequest(InputModel):
    action: Literal["heart"]
    age: Annotated[int, Field(ge=20, le=100)]
    sex: Annotated[int, Field(ge=0, le=1)]
    cp: Annotated[int, Field(ge=0, le=3)]
    trestbps: Annotated[int, Field(ge=80, le=200)]
    chol: Annotated[int, Field(ge=100, le=600)]
    fbs: Annotated[int, Field(ge=0, le=1)]
    restecg: Annotated[int, Field(ge=0, le=2)]
    thalach: Annotated[int, Field(ge=60, le=220)]
    exang: Annotated[int, Field(ge=0, le=1)]
    oldpeak: Annotated[float, Field(ge=0, le=10)]
    slope: Annotated[int, Field(ge=0, le=2)]
    ca: Annotated[int, Field(ge=0, le=4)]
    thal: Annotated[int, Field(ge=0, le=3)]


PredictionRequest = Annotated[
    DiabetesRequest | HeartRequest,
    Field(discriminator="action"),
]


@lru_cache(maxsize=4)
def load_model(filename: str):
    model_path = MODEL_DIR / filename
    if not model_path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Required model artifact is missing: {filename}",
        )
    return joblib.load(model_path)


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/")
def predict(request: PredictionRequest):
    try:
        if isinstance(request, DiabetesRequest):
            model = load_model("diabetes_model.pkl")
            values = np.asarray(
                [[
                    request.pregnancies,
                    request.glucose,
                    request.blood_pressure,
                    request.skin_thickness,
                    request.insulin,
                    request.bmi,
                    request.dpf,
                    request.age,
                ]],
                dtype=float,
            )
            prediction = int(model.predict(values)[0])
            diabetic_probability = float(model.predict_proba(values)[0][1])
            return {
                "type": "diabetes",
                "prediction": prediction,
                "diabetic_probability": diabetic_probability,
                "healthy_probability": 1 - diabetic_probability,
                "risk": "high" if prediction == 1 else "low",
            }

        scaler = load_model("scaler.pkl")
        model = load_model("kmeans_model.pkl")
        numeric_values = np.asarray(
            [[
                request.age,
                request.trestbps,
                request.chol,
                request.thalach,
                request.oldpeak,
            ]],
            dtype=float,
        )
        scaled_values = scaler.transform(numeric_values)[0]
        selected_values = np.asarray(
            [[
                request.sex,
                scaled_values[3],
                request.exang,
                scaled_values[4],
                request.ca,
                int(request.cp == 2),
                int(request.slope == 1),
                int(request.slope == 2),
                int(request.thal == 2),
                int(request.thal == 3),
            ]],
            dtype=float,
        )
        cluster = int(model.predict(selected_values)[0])
        cluster_info = {
            0: {
                "label": "Low Risk Profile",
                "description": (
                    "The model assigned this patient to cluster 0. "
                    "Continue routine check-ups."
                ),
                "recommendation": (
                    "Maintain a heart-healthy diet and regular exercise routine."
                ),
            },
            1: {
                "label": "Moderate Risk / At-Risk",
                "description": (
                    "The model assigned this patient to cluster 1. "
                    "Closer monitoring may be appropriate."
                ),
                "recommendation": (
                    "Discuss follow-up monitoring with a qualified healthcare professional."
                ),
            },
            2: {
                "label": "High Risk Profile",
                "description": (
                    "The model assigned this patient to cluster 2. "
                    "Further clinical assessment may be appropriate."
                ),
                "recommendation": (
                    "Consult a qualified healthcare professional about further assessment."
                ),
            },
        }
        return {
            "type": "heart",
            "cluster": cluster,
            **cluster_info.get(
                cluster,
                {
                    "label": f"Cluster {cluster}",
                    "description": "No interpretation is defined for this cluster.",
                    "recommendation": (
                        "Consult a qualified healthcare professional."
                    ),
                },
            ),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the deployment logs for details.",
        ) from exc
