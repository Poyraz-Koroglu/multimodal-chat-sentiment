import torch
from torch.utils.data import Dataset
import librosa
import pandas as pd
from transformers import AutoTokenizer


class SpeechTextDataset(Dataset):
    def __init__(self, excel_path, sample_rate=16000, transform=None, target_seconds=4, tokenizer_name="bert-base-uncased", max_length = 128):
        """
        Args:
            excel_path (str): Path to Excel file with columns [Audio, Label, Transcript]
            sample_rate (int): Desired sample rate for audio
            transform (callable, optional): Optional torchaudio transforms
            target_seconds (int): Target audio length in seconds
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.df = pd.read_excel(excel_path)
        self.sample_rate = sample_rate
        self.transform = transform
        self.excel_path = excel_path
        self.target_seconds = target_seconds
        self.label_map = None

        labels = sorted(self.df["Label"].unique())
        self.label_map = {}
        if "neutral" in labels:
            self.label_map["neutral"] = 0
        if "negative" in labels:
            self.label_map["negative"] = 1

        self.df["length"] = self.df["Audio"].apply(lambda path: librosa.get_duration(filename=path))
        self.df = self.df.sort_values("length").reset_index(drop=True)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio_np, sr = librosa.load(row["Audio"], sr=self.sample_rate)
        audio_tensor = torch.tensor(audio_np).unsqueeze(0)
        target_samples = int(self.sample_rate * self.target_seconds)
        n_samples = audio_tensor.shape[1]


        if n_samples < target_samples:
            repeats = target_samples // n_samples
            remainder = target_samples % n_samples
            if repeats > 1:
                repeats = 1
            audio_tensor = torch.cat([audio_tensor, audio_tensor[:, :remainder:]],dim=1)
            if repeats >0:
                audio_tensor = torch.cat([audio_tensor[:, :remainder:], audio_tensor[:, remainder:]], dim=1)
        elif n_samples > target_samples:
            audio_tensor = audio_tensor[:, :target_samples]

        if self.transform:
            audio_tensor = self.transform(audio_tensor)

        label_id = self.label_map[row["Label"]]

        transcript = str(row["Transcript"])

        encoded = self.tokenizer(
            transcript,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        return {
            "audio": audio_tensor,
            "label": torch.tensor(label_id, dtype=torch.long),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "transcript": transcript,
        }
    """
dataset = SpeechTextDataset(excel_path="C:/Users/poyraz.koroglu/Desktop/Custom-Dataset-Test.xlsx",
                                sample_rate=16000,
                                transform=None)
sample = dataset[0]
print(sample.keys())
print(sample['audio'].shape)
print(sample['label'])
    """