import torch
import torch.nn as nn
from transformers import WhisperModel, DistilBertModel, DistilBertTokenizer, WhisperFeatureExtractor
from typing import Optional

# Import your custom ProjectLogger
from logger import ProjectLogger

class SpeechTextModel(nn.Module):
    """
    Multimodal model combining Whisper (audio) and DistilBERT (text)
    for sentiment classification.
    """

    def __init__(
        self,
        logger_: Optional[object] = None,
        whisper_model_name: str = "openai/whisper-small",
        distilbert_model_name: str = "distilbert-base-uncased",
        freeze_whisper: bool = True,
        freeze_distilbert: bool = True,
        num_classes: int = 2,
        dropout: float = 0.3,
        hidden_dim: int = 768
    ):
        super().__init__()
        # --- Logger setup ---
        self.logger = logger_ or ProjectLogger("backend").get_logger()
        self.logger.info(f"Initializing SpeechTextModel with Whisper={whisper_model_name}, DistilBERT={distilbert_model_name}")

        # --- Model attributes ---
        self.num_classes = num_classes
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        # --- Whisper model + feature extractor ---
        self.whisper = WhisperModel.from_pretrained(whisper_model_name)
        self.audio_processor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)
        self.whisper_dim = self.whisper.config.hidden_size

        if freeze_whisper:
            for param in self.whisper.parameters():
                param.requires_grad = False
            self.logger.info("Whisper model frozen.")

        # --- DistilBERT model ---
        self.distilbert = DistilBertModel.from_pretrained(distilbert_model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_name)
        self.distilbert_dim = self.distilbert.config.hidden_size

        if freeze_distilbert:
            for param in self.distilbert.parameters():
                param.requires_grad = False
            self.logger.info("DistilBERT model frozen.")

        # --- Classification heads ---
        self._initiate_heads(hidden_dim)
        self.logger.info(f"SpeechTextModel initialized with hidden_dim={hidden_dim}, dropout={dropout}, num_classes={num_classes}")

    # --- Audio feature extraction ---
    def _extract_audio_features(self, audio: torch.Tensor) -> torch.Tensor:
        self.logger.debug(f"Extracting audio features from tensor with shape {audio.shape}")
        audio = audio.squeeze(1).cpu().numpy()
        inputs = self.audio_processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        inputs = inputs.to(next(self.whisper.parameters()).device)
        encoder_outputs = self.whisper.encoder(inputs, return_dict=True)
        features = encoder_outputs.last_hidden_state.mean(dim=1)
        self.logger.debug(f"Extracted audio features with shape {features.shape}")
        return features

    # --- Text feature extraction ---
    def _extract_text_features(self, input_ids, attention_mask) -> torch.Tensor:
        self.logger.debug(f"Extracting text features with input_ids shape {input_ids.shape}")
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        features = outputs.last_hidden_state[:, 0, :]
        self.logger.debug(f"Extracted text features with shape {features.shape}")
        return features

    # --- Initialize classification heads ---
    def _initiate_heads(self, hidden_dim: int):
        self.audio_head = nn.Sequential(
            nn.Linear(self.whisper_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )
        self.text_head = nn.Sequential(
            nn.Linear(self.distilbert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout * 0.5),
            nn.Linear(hidden_dim // 2, self.num_classes),
        )
        self._initialize_weights()
        self.logger.info("Classification heads initialized.")

    def _initialize_weights(self):
        for module in [self.audio_head, self.text_head, self.fusion_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
        self.logger.debug("Weights initialized using Xavier uniform.")

    def _fuse_features(self, audio_features, text_features):
        fused = torch.cat([audio_features, text_features], dim=1)
        self.logger.debug(f"Fused features shape: {fused.shape}")
        return fused

    # --- Forward pass ---
    def forward(self, audio, input_ids, attention_mask):
        self.logger.debug("Running forward pass")
        audio_features = self._extract_audio_features(audio)
        text_features = self._extract_text_features(input_ids, attention_mask)

        audio_features = self.audio_head(audio_features)
        text_features = self.text_head(text_features)

        fused = self._fuse_features(audio_features, text_features)
        logits = self.fusion_head(fused)

        self.logger.info(f"Forward pass complete. Logits shape: {logits.shape}")
        return {
            "logits": logits,
            "audio_features": audio_features,
            "text_features": text_features,
            "fused_features": fused,
        }

    # --- Save / Load ---
    def save_model(self, path: str):
        torch.save(self.state_dict(), path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.logger.info(f"Model loaded from {path}")

    def get_model_info(self):
        info = {
            "whisper_dim": self.whisper_dim,
            "distilbert_dim": self.distilbert_dim,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "frozen_whisper": all(not p.requires_grad for p in self.whisper.parameters()),
            "frozen_distilbert": all(not p.requires_grad for p in self.distilbert.parameters()),
        }
        self.logger.info(f"Model info: {info}")
        return info
