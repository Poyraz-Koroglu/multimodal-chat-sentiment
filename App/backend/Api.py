import os
import io
import torch
import librosa
import datetime
from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional, Union
from elasticsearch import Elasticsearch

import logger
from logger import ProjectLogger
from SpeechTextDataset import SpeechTextDataset
from Model import SpeechTextModel
from fastapi.middleware.cors import CORSMiddleware



logger_ = ProjectLogger("backend").get_logger()

app = FastAPI(title="Multimodal Sentiment Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load trained model ---
MODEL_PATH = "../../Saved_Models/fastapi_trial_model.pth"
model = SpeechTextModel(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --- Dummy dataset just for tokenizer & preprocessing ---
dummy_dataset = SpeechTextDataset(
    excel_path=os.path.join("../../Data", "dummy.xlsx"),
    validate_files=False
)

# --- Elasticsearch connection ---
es = Elasticsearch("http://elasticsearch:9200")

@app.post("/predict")
async def predict(
    text: Optional[str] = Form(None),
    audio: Optional[Union[UploadFile, str]] = Form(None)
):
    """
    Predict sentiment from text and/or audio.
    Returns: 0 for neutral, 1 for negative
    """
    logger_.info(f"Received /predict request | text_provided={bool(text)} | audio_provided={bool(audio)}")
    if not text and not audio:
        logger_.warning("No text or audio provided")
        return {"error": "Please provide either text or audio."}

    # --- Process text ---
    if text:
        logger_.info("Processing Text Provided")
        encoded = dummy_dataset.tokenizer(
            text,
            max_length=dummy_dataset.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        logger_.debug(f"Text input_ids shape: {input_ids.shape}")

    else:
        input_ids = None
        attention_mask = None
        logger_.info("Input ID's and Attention Mask Not Provided")

    # --- Process audio ---
    logger_.info("Processing Audio")
    if isinstance(audio, str) or audio is None or audio == "":
        logger_.info("No audio provided, using dummy zeros tensor")
        audio_tensor = torch.zeros(1, 16000*7)
    else:
        try:
            audio_bytes = await audio.read()
            audio_np, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
            audio_tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
            audio_tensor = dummy_dataset.normalize_audio_length_dynamic(audio_tensor)
            audio_tensor = audio_tensor.to(device)
            logger_.debug(f"Audio tensor shape after preprocessing: {audio_tensor.shape}")
        except Exception as e:
            logger_.exception(f"Error while preprocessing audio {e}")
            return {"error": f"Error while preprocessing audio "}
    # --- Run inference ---
    with torch.no_grad():
        outputs = model(audio_tensor, input_ids, attention_mask)
        sentiment = torch.argmax(outputs["logits"], dim=1).item()

    # --- Log prediction to Elasticsearch ---
    log_doc = {
        "timestamp": datetime.datetime.utcnow(),
        "text": text if text else None,
        "audio_provided": bool(audio),
        "prediction": sentiment
    }
    try:
        es.index(index="ml-logs", document=log_doc)
        logger_.info("Prediction successfully logged to Elasticsearch")
    except Exception as e:
        logger_.error(f"Elasticsearch logging failed: {e}")


    logger_.info("Sentiment Prediction Complete")
    return {"prediction": sentiment}
