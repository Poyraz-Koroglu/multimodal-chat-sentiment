import torch
from torch.utils.data import Dataset
import librosa
import pandas as pd
from transformers import AutoTokenizer
import os
import logging
from typing import Optional, Callable, Dict, Any


class SpeechTextDataset(Dataset):
    def __init__(self,
                 excel_path: str="C:/Users/poyraz.koroglu/Desktop/Trial-Dataset-root/metadata.cs",
                 sample_rate: int = 16000,
                 transform: Optional[Callable] = None,
                 target_seconds: float = 8.0,
                 tokenizer_name: str = "bert-base-uncased",
                 max_length: int = 128,

                 validate_files: bool = True):
        """
        Args:
            excel_path (str): Path to Excel file with columns [Audio, Label, Transcript]
            sample_rate (int): Desired sample rate for audio
            transform (callable, optional): Optional torchaudio transforms
            target_seconds (float): Target audio length in seconds
            tokenizer_name (str): HuggingFace tokenizer name
            max_length (int): Maximum sequence length for tokenization
            validate_files (bool): Whether to validate audio files exist during init
        """
        self.sample_rate = sample_rate
        self.transform = transform
        self.target_seconds = target_seconds
        self.target_samples = int(sample_rate * target_seconds)
        self.max_length = max_length
        self.excel_path = excel_path
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer '{tokenizer_name}': {e}")

        # Load dataset
        try:
            if excel_path.endswith(".csv"):
                self.df = pd.read_excel(excel_path)
            elif excel_path.endswith(".xlsx") or excel_path.endswith(".xls"):
                self.df = pd.read_excel(excel_path, engine="openpyxl")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load Excel file '{excel_path}': {e}")

        # Validate required columns
        required_columns = ["Audio", "Label", "Transcript"]
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Clean data
        self._clean_data()

        # Create binary label mapping (negative/neutral only)
        self._create_binary_label_mapping()

        # Validate files if requested
        if validate_files:
            self._validate_audio_files()

        # ALWAYS compute lengths and sort by length (smallest to largest)
        logging.info("Computing audio lengths and sorting dataset...")
        self._compute_and_sort_by_length()

        logging.info(f"Dataset loaded: {len(self.df)} samples, sorted by audio length")

    def _clean_data(self):
        """Clean the dataset by removing invalid entries."""
        initial_len = len(self.df)

        # Remove rows with missing values
        self.df = self.df.dropna(subset=["Audio", "Label", "Transcript"])

        # Convert transcript to string and handle empty transcripts
        self.df["Transcript"] = self.df["Transcript"].astype(str)
        self.df = self.df[self.df["Transcript"].str.strip() != ""]

        # Ensure only negative/neutral labels
        valid_labels = {"negative", "neutral"}
        invalid_labels = set(self.df["Label"].unique()) - valid_labels
        if invalid_labels:
            logging.warning(f"Removing {len(invalid_labels)} invalid label types: {invalid_labels}")
            self.df = self.df[self.df["Label"].isin(valid_labels)]

        final_len = len(self.df)
        if final_len < initial_len:
            logging.warning(f"Removed {initial_len - final_len} invalid entries")

        if len(self.df) == 0:
            raise ValueError("No valid samples remaining after cleaning")

        self.df = self.df.reset_index(drop=True)

    def _create_binary_label_mapping(self):
        """Create binary label mapping: neutral=0, negative=1."""
        unique_labels = set(self.df["Label"].unique())

        # Ensure we only have neutral and/or negative
        valid_labels = {"negative", "neutral"}
        if not unique_labels.issubset(valid_labels):
            invalid = unique_labels - valid_labels
            raise ValueError(f"Dataset contains invalid labels: {invalid}. Only 'negative' and 'neutral' are allowed.")

        # Fixed binary mapping
        self.label_map = {"neutral": 0, "negative": 1}
        self.num_classes = 2

        logging.info(f"Binary label mapping: {self.label_map}")
        label_counts = self.df["Label"].value_counts().to_dict()
        logging.info(f"Label distribution: {label_counts}")

    def _validate_audio_files(self):
        """Validate that all audio files exist."""
        missing_files = []
        for idx, audio_path in enumerate(self.df["Audio"]):
            if not os.path.exists(audio_path):
                missing_files.append((idx, audio_path))

        if missing_files:
            logging.error(f"Found {len(missing_files)} missing audio files")
            for idx, path in missing_files[:5]:  # Show first 5
                logging.error(f"  Row {idx}: {path}")
            if len(missing_files) > 5:
                logging.error(f"  ... and {len(missing_files) - 5} more")
            raise FileNotFoundError(f"Missing {len(missing_files)} audio files")

    def _compute_and_sort_by_length(self):
        """Compute audio lengths and sort dataset from smallest to largest."""

        def get_duration_safe(path):
            try:
                return librosa.get_duration(filename=path)
            except Exception as e:
                logging.warning(f"Failed to get duration for {path}: {e}")
                return float('inf')  # Put problematic files at the end

        logging.info("Computing audio durations...")
        self.df["length"] = self.df["Audio"].apply(get_duration_safe)

        # Sort by length (smallest to largest)
        self.df = self.df.sort_values("length").reset_index(drop=True)

        # Log statistics
        valid_lengths = self.df[self.df["length"] != float('inf')]["length"]
        if len(valid_lengths) > 0:
            logging.info(f"Audio length stats - Min: {valid_lengths.min():.2f}s, "
                         f"Max: {valid_lengths.max():.2f}s, "
                         f"Mean: {valid_lengths.mean():.2f}s, "
                         f"Target: {self.target_seconds}s")

        # Check for problematic files
        problematic = self.df[self.df["length"] == float('inf')]
        if len(problematic) > 0:
            logging.warning(f"Found {len(problematic)} files with duration issues")

    import torch

    def normalize_audio_length_dynamic(audio: torch.Tensor, sample_rate: int, target_seconds: float = 7.0,
                                       short_threshold: float = 2.0):

        target_samples = int(target_seconds * sample_rate)
        n_samples = audio.shape[-1]
        audio_seconds = n_samples / sample_rate

        if audio_seconds <= short_threshold:
            # Repeat until half target, then pad with zeros
            half_target_samples = target_samples // 2
            repeats = half_target_samples // n_samples
            remainder = half_target_samples % n_samples

            audio_repeated = audio.repeat(1, repeats)
            if remainder > 0:
                audio_repeated = torch.cat([audio_repeated, audio[:, :remainder]], dim=1)

            remaining_samples = target_samples - audio_repeated.shape[-1]
            zero_padding = torch.zeros(audio.shape[0], remaining_samples)
            audio_normalized = torch.cat([audio_repeated, zero_padding], dim=1)

        else:
            # Random choice: either pad with zeros or repeat
            import random
            if random.random() < 0.5:
                # Zero-pad
                pad_samples = target_samples - n_samples
                if pad_samples > 0:
                    zero_padding = torch.zeros(audio.shape[0], pad_samples)
                    audio_normalized = torch.cat([audio, zero_padding], dim=1)
                else:
                    audio_normalized = audio[:, :target_samples]
            else:
                # Repeat
                repeats = target_samples // n_samples
                remainder = target_samples % n_samples
                audio_normalized = audio.repeat(1, repeats)
                if remainder > 0:
                    audio_normalized = torch.cat([audio_normalized, audio[:, :remainder]], dim=1)

        return audio_normalized

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            row = self.df.iloc[idx]

            # Load audio
            try:
                audio_np, sr = librosa.load(row["Audio"], sr=self.sample_rate)
                if len(audio_np) == 0:
                    raise ValueError("Empty audio file")
            except Exception as e:
                logging.error(f"Failed to load audio {row['Audio']}: {e}")
                raise RuntimeError(f"Failed to load audio file at index {idx}: {e}")

            # Convert to tensor and normalize length
            audio_tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
            audio_tensor = self.normalize_audio_length_dynamic(audio_tensor)

            # Apply transforms
            if self.transform:
                audio_tensor = self.transform(audio_tensor)

            # Get binary label (guaranteed to be negative or neutral)
            label_str = row["Label"]
            label_id = self.label_map[label_str]  # Will be 0 for neutral, 1 for negative

            # Process transcript
            transcript = str(row["Transcript"]).strip()
            if not transcript:
                logging.warning(f"Empty transcript at index {idx}")
                transcript = "[EMPTY]"  # Fallback

            # Tokenize transcript
            try:
                encoded = self.tokenizer(
                    transcript,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].squeeze(0)
                attention_mask = encoded["attention_mask"].squeeze(0)
            except Exception as e:
                logging.error(f"Failed to tokenize transcript at index {idx}: {e}")
                raise RuntimeError(f"Tokenization failed at index {idx}: {e}")

            return {
                "audio": audio_tensor,
                "label": torch.tensor(label_id, dtype=torch.long),
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "transcript": transcript,
            }

        except Exception as e:
            logging.error(f"Error processing sample {idx}: {e}")
            raise

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced binary classification."""
        label_counts = self.df["Label"].value_counts()

        neutral_count = label_counts.get("neutral", 0)
        negative_count = label_counts.get("negative", 0)
        total_samples = len(self.df)

        # Compute inverse frequency weights
        neutral_weight = total_samples / (2 * neutral_count) if neutral_count > 0 else 0.0
        negative_weight = total_samples / (2 * negative_count) if negative_count > 0 else 0.0

        # Return weights in order: [neutral, negative]
        return torch.tensor([neutral_weight, negative_weight], dtype=torch.float32)

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        label_counts = self.df["Label"].value_counts().to_dict()

        # Audio length stats
        valid_lengths = self.df[self.df["length"] != float('inf')]["length"]
        audio_stats = {}
        if len(valid_lengths) > 0:
            audio_stats = {
                "min_duration": f"{valid_lengths.min():.2f}s",
                "max_duration": f"{valid_lengths.max():.2f}s",
                "mean_duration": f"{valid_lengths.mean():.2f}s",
                "std_duration": f"{valid_lengths.std():.2f}s",
            }

        return {
            "total_samples": len(self.df),
            "label_distribution": label_counts,
            "class_balance": {
                "neutral_ratio": label_counts.get("neutral", 0) / len(self.df),
                "negative_ratio": label_counts.get("negative", 0) / len(self.df)
            },
            "audio_stats": audio_stats,
            "target_audio_length": f"{self.target_seconds}s ({self.target_samples} samples)",
            "sample_rate": self.sample_rate,
            "tokenizer": self.tokenizer.name_or_path,
            "max_sequence_length": self.max_length,
            "data_order": "Sorted by audio length (shortest to longest)"
        }

    def get_length_at_index(self, idx: int) -> float:
        """Get the original audio length for a given index."""
        return self.df.iloc[idx]["length"]