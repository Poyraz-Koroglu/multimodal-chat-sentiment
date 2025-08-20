# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Transformers (for Whisper / DistilBERT)
from transformers import DistilBertTokenizer, WhisperFeatureExtractor

# Your custom classes
from Model import SpeechTextModel
from SpeechTextDataset import SpeechTextDataset
from Train import train_model
# Standard Python
import os
import logging
from typing import Any, Dict

device = "cuda" if torch.cuda.is_available() else "cpu"


trainDs = SpeechTextDataset("C:/Users/poyraz.koroglu/Desktop/Trial-Dataset-root/metadata.csv")
trainDataloader = DataLoader(trainDs, batch_size=32, shuffle=True)
valDs = SpeechTextDataset("C:/Users/poyraz.koroglu/Desktop/Trial-Dataset-root/metadata.csv")
valDataloader = DataLoader(valDs, batch_size=32, shuffle=True)

model = SpeechTextModel()
num_epochs = 10

train_model(model, trainDataloader, num_epochs, device)

