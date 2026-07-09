import os
import sys
import certifi
import pymongo
import pandas as pd

from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from uvicorn import run as app_run

from Networksecurity.exception.exception import NetworkSecurityException
from Networksecurity.logging.logger import logging
from Networksecurity.pipeline.training_pipeline import TrainingPipeline
from Networksecurity.utils.main_utils.utils import load_object
from Networksecurity.utils.ml_utils.model.estimator import NetworkModel

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------

load_dotenv()

mongo_db_url = os.getenv("MONGO_DB_URI")

ca = certifi.where()

client = pymongo.MongoClient(
    mongo_db_url,
    tlsCAFile=ca
)

from Networksecurity.constant.training_pipeline import (
    DATA_INGESTION_DATABASE_NAME,
    DATA_INGESTION_COLLECTION_NAME,
)

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# ------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

MODEL_DIR = "final_models"

PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "preprocessor.pkl"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "model.pkl"
)

OUTPUT_DIR = "predicted_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Home
# ------------------------------------------------------------------


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------


@app.get("/train")
async def train_route():

    try:

        logging.info("Training started")

        train_pipeline = TrainingPipeline()

        train_pipeline.run_pipeline()

        return Response(
            content="Training completed successfully.",
            media_type="text/plain"
        )

    except Exception as e:

        logging.exception(e)

        raise NetworkSecurityException(e, sys)


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------


@app.post("/predict")
async def predict_route(
    request: Request,
    file: UploadFile = File(...)
):

    try:

        # ------------------------------------------------------
        # Read CSV
        # ------------------------------------------------------

        df = pd.read_csv(file.file)

        logging.info(f"Uploaded Shape : {df.shape}")

        # ------------------------------------------------------
        # Remove unwanted columns
        # ------------------------------------------------------

        drop_cols = []

        for col in ["Index", "Unnamed: 0"]:

            if col in df.columns:
                drop_cols.append(col)

        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

        # ------------------------------------------------------
        # Remove target column if uploaded
        # ------------------------------------------------------

        target_columns = [
            "Result",
            "result",
            "class",
            "Class",
            "target",
            "Target",
            "label",
            "Label",
            "y",
            "Y"
        ]

        for col in target_columns:

            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # ------------------------------------------------------
        # Load models
        # ------------------------------------------------------

        if not os.path.exists(PREPROCESSOR_PATH):
            raise Exception(
                "preprocessor.pkl not found. Train model first."
            )

        if not os.path.exists(MODEL_PATH):
            raise Exception(
                "model.pkl not found. Train model first."
            )

        preprocessor = load_object(PREPROCESSOR_PATH)

        final_model = load_object(MODEL_PATH)

        # ------------------------------------------------------
        # Validate feature names
        # ------------------------------------------------------

        if hasattr(preprocessor, "feature_names_in_"):

            expected_columns = list(preprocessor.feature_names_in_)

            missing_columns = list(
                set(expected_columns) - set(df.columns)
            )

            if len(missing_columns) > 0:

                raise Exception(
                    f"Missing Columns : {missing_columns}"
                )

            extra_columns = list(
                set(df.columns) - set(expected_columns)
            )

            if len(extra_columns) > 0:
                df.drop(columns=extra_columns, inplace=True)

            df = df[expected_columns]

        # ------------------------------------------------------
        # Prediction
        # ------------------------------------------------------

        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=final_model
        )

        predictions = network_model.predict(df)

        df["Prediction"] = predictions

        # ------------------------------------------------------
        # Save Output
        # ------------------------------------------------------

        output_path = os.path.join(
            OUTPUT_DIR,
            "output.csv"
        )

        df.to_csv(
            output_path,
            index=False
        )

        logging.info(f"Prediction saved : {output_path}")

        # ------------------------------------------------------
        # JSON Response
        # ------------------------------------------------------

        if "application/json" in request.headers.get(
            "accept",
            ""
        ):

            return {
                "status": "success",
                "rows": len(df),
                "prediction_file": output_path,
                "predictions": predictions.tolist()
            }

        # ------------------------------------------------------
        # Browser Response
        # ------------------------------------------------------

        table = df.to_html(
            classes="table table-striped",
            index=False
        )

        return templates.TemplateResponse(
            "table.html",
            {
                "request": request,
                "table": table
            }
        )

    except Exception as e:

        logging.exception(e)

        raise NetworkSecurityException(e, sys)


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------

if __name__ == "__main__":

    app_run(
        app,
        host="127.0.0.1",
        port=8000
    )