import torch
from torch.utils.data import Dataset
import librosa
import pandas as pd


class SpeechTextDataset(Dataset):
    def __init__(self, excel_path, sample_rate=16000, transform=None):
        """
        Args:
            excel_path (str): Path to Excel file with columns [Audio, Label]
            sample_rate (int): Desired sample rate for audio
            transform (callable, optional): Optional torchaudio transforms
        """
        self.df = pd.read_excel(excel_path)
        self.sample_rate = sample_rate
        self.transform = transform
        self.excel_path = excel_path

        unique_labels = sorted(self.df["Label"].unique())
        self.label_map = {}
        if "neutral" in unique_labels:
            self.label_map["neutral"] = 0
        if "negative" in unique_labels:
            self.label_map["negative"] = 1


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio_np, sr = librosa.load(row["Audio"], sr=self.sample_rate)  # sr=self.sample_rate resamples automatically
        audio_tensor = torch.tensor(audio_np).unsqueeze(0)
        length_seconds = audio_tensor.shape[1] / sr
        target_seconds = 4
        target_samples = int(self.sample_rate * target_seconds)

        if audio_tensor.shape[1] > target_samples:
            audio_tensor = audio_tensor[:, :target_samples]
        elif audio_tensor.shape[1] < target_samples:
            padding = target_samples - audio_tensor.shape[1]
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))


        if self.transform:
            audio_tensor = self.transform(audio_tensor)

        label_id = self.label_map[row["Label"]]

        return {
            "audio": audio_tensor,
            "label": torch.tensor(label_id, dtype=torch.long)
        }
dataset = SpeechTextDataset(excel_path="C:/Users/poyraz.koroglu/Desktop/Custom-Dataset-Test.xlsx",
                                sample_rate=16000,
                                transform=None)
sample = dataset[0]
print(sample.keys())
print(sample['audio'].shape)
print(sample['label'])
