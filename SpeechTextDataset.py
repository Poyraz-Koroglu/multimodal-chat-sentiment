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

        labels = sorted(self.df["Label"].unique())
        self.label_map = {}
        if "neutral" in labels:
            self.label_map["neutral"] = 0
        if "negative" in labels:
            self.label_map["negative"] = 1


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio_np, sr = librosa.load(row["Audio"], sr=self.sample_rate)
        audio_tensor = torch.tensor(audio_np).unsqueeze(0)
        target_seconds = 4
        target_samples = int(self.sample_rate * target_seconds)
        n_samples = audio_np.shape[0].shape[1]


        if n_samples < target_samples:
            repeats = target_samples // n_samples
            remainder = target_samples % n_samples
            audio_tensor = audio_tensor.repeat(1, repeats)  # repeat along time axis
            audio_tensor = torch.cat([audio_tensor, audio_tensor[:, :remainder]], dim=1)
        elif n_samples > target_samples:
            audio_tensor = audio_tensor[:, :target_samples]

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
