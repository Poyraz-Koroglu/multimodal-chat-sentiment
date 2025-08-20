import torch
import torch.nn as nn
from transformers import (
    WhisperModel,
    DistilBertModel,
    DistilBertTokenizer,
    WhisperFeatureExtractor,
)
from datasets import load_dataset
import torchaudio
import logging


class SpeechTextModel(nn.Module):
    """
    SpeechText: Multimodal Speech-Text Model
    Combines Whisper (audio) and DistilBERT (text) embeddings for classification tasks.
    """

    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-small",
        distilbert_model_name: str = "distilbert-base-uncased",
        freeze_whisper: bool = True,
        freeze_distilbert: bool = True,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout

        # Whisper model + feature extractor
        self.whisper = WhisperModel.from_pretrained(whisper_model_name)
        self.audio_processor = WhisperFeatureExtractor.from_pretrained(
            whisper_model_name
        )

        if freeze_whisper:
            for param in self.whisper.parameters():
                param.requires_grad = False

        self.whisper_dim = self.whisper.config.d_model

        # DistilBERT model
        self.distilbert = DistilBertModel.from_pretrained(distilbert_model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_name)

        if freeze_distilbert:
            for param in self.distilbert.parameters():
                param.requires_grad = False

        self.distilbert_dim = self.distilbert.config.hidden_size

        # Fusion + classifier
        self._create_fusion_layers()

        logging.info(
            f"Initialized WhisBERT with Whisper={whisper_model_name}, "
            f"DistilBERT={distilbert_model_name}, "
            f"freeze_whisper={freeze_whisper}, freeze_distilbert={freeze_distilbert}"
        )

    # Audio feature extraction
    def _extract_audio_features(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [B, 1, samples]
        audio = audio.squeeze(1).cpu().numpy()
        inputs = self.audio_processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features
        inputs = inputs.to(next(self.whisper.parameters()).device)
        encoder_outputs = self.whisper.encoder(inputs, return_dict=True)
        return encoder_outputs.last_hidden_state.mean(dim=1)  # [B, whisper_dim]

    # Text feature extraction
    def _extract_text_features(self, input_ids, attention_mask) -> torch.Tensor:
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [B, distilbert_dim]

    # Fusion layers
    def _create_fusion_layers(self):
        fusion_dim = self.whisper_dim + self.distilbert_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, self.num_classes),
        )

    def _fuse_features(self, audio_features, text_features):
        return torch.cat([audio_features, text_features], dim=1)

    # Forward pass
    def forward(self, audio, input_ids, attention_mask):
        audio_features = self._extract_audio_features(audio)
        text_features = self._extract_text_features(input_ids, attention_mask)

        fused = self._fuse_features(audio_features, text_features)
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "audio_features": audio_features,
            "text_features": text_features,
            "fused_features": fused,
        }

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)
        logging.info(f"Model saved to {path}")

    def load_model(self, path: str, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
        logging.info(f"Model loaded from {path}")

    def get_model_info(self):
        return {
            "whisper_dim": self.whisper_dim,
            "distilbert_dim": self.distilbert_dim,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "frozen_whisper": all(
                not p.requires_grad for p in self.whisper.parameters()
            ),
            "frozen_distilbert": all(
                not p.requires_grad for p in self.distilbert.parameters()
            ),
        }
        #############
        ##UNIT TEST##
        ####AUDIO####
        #############

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy",
    "clean",
    split="validation"
)


# Get file path directly
file_path = ds[0]["file"]  # get the path to the audio file
waveform, sr = torchaudio.load(file_path)  # load audio manually

# Add batch & channel dimensions
audio_tensor = waveform.unsqueeze(0)  # [B, 1, seq_len]

model = SpeechTextModel()
with torch.no_grad():
    audio_features = model._extract_audio_features(audio_tensor)

print("Extracted Whisper features:", audio_features.shape)
####UNIT TEST####
#####TEXT########
"""

input = "I am so angry!"

model = SpeechTextModel()
tokenizer = model.tokenizer

encoding = tokenizer(
    input,
    return_tensors="pt",   # returns PyTorch tensors
    truncation=True,
    padding="max_length",
    max_length=32          # or whatever length you want
)

input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]


with torch.no_grad():
    text_features = model._extract_text_features(input_ids, attention_mask)
print("Extracted Whisper features:", text_features.shape)
"""
