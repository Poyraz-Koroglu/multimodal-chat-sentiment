import os
import logging
from typing import Optional, Callable, Dict, Any

# Import torch first and check for issues
try:
    import torch
    from torch.utils.data import Dataset
except ImportError as e:
    raise ImportError(f"Failed to import torch: {e}")

# Import other libraries
import librosa
import pandas as pd

# Import transformers separately with error handling
try:
    from transformers import AutoTokenizer
except ImportError as e:
    raise ImportError(f"Failed to import transformers: {e}. Please install with: pip install transformers")
import random
from nltk.corpus import wordnet
import nltk
import numpy as np

def synonym_replacement(sentence, n=1):
    """Replace up to n words in the sentence with synonyms."""
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    num_replaced = 0

    for word in random_word_list:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words = [synonym if w == word else w for w in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_words)


def random_deletion(sentence, p=0.1):
    """Randomly delete words from the sentence with probability p."""
    words = sentence.split()
    if len(words) == 1:
        return sentence
    new_words = [w for w in words if random.random() > p]
    if len(new_words) == 0:
        return random.choice(words)
    return ' '.join(new_words)

def time_stretch_audio(audio_tensor: torch.Tensor, min_rate=0.8, max_rate=1.2) -> torch.Tensor:
    """
    Randomly time-stretch an audio tensor.

    Args:
        audio_tensor: Tensor of shape [1, samples]
        min_rate: Minimum stretch factor (<1 slows down)
        max_rate: Maximum stretch factor (>1 speeds up)

    Returns:
        Time-stretched audio tensor of shape [1, samples]
    """
    rate = random.uniform(min_rate, max_rate)  # random stretch factor
    audio_np = audio_tensor.squeeze(0).cpu().numpy()
    stretched = librosa.effects.time_stretch(audio_np, rate=rate)
    stretched_tensor = torch.tensor(stretched, dtype=torch.float32).unsqueeze(0)
    # Convert back to tensor and ensure it has same shape (pad or crop)
    stretched_tensor = torch.tensor(stretched, dtype=torch.float32).unsqueeze(0)
    if stretched_tensor.shape[1] < audio_tensor.shape[1]:
        # pad
        pad_size = audio_tensor.shape[1] - stretched_tensor.shape[1]
        stretched_tensor = torch.nn.functional.pad(stretched_tensor, (0, pad_size))
    else:
        # crop
        stretched_tensor = stretched_tensor[:, :audio_tensor.shape[1]]

    return stretched_tensor

class SpeechTextDataset(Dataset):
    def __init__(self,
                 excel_path: str,
                 sample_rate: int = 16000,
                 transform: Optional[Callable] = None,
                 target_seconds: float = 8.0,
                 tokenizer_name: str = "bert-base-uncased",
                 max_length: int = 128,
                 validate_files: bool = True,
                 augment: bool = True,
                 augment_prob = 0.3):
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
        logger_ = ProjectLogger("dataset").get_logger()
        self.sample_rate = sample_rate
        self.transform = transform
        self.target_seconds = target_seconds
        self.target_samples = int(sample_rate * target_seconds)
        self.max_length = max_length
        self.excel_path = excel_path
        logger_.info(f)
        self.augment = augment
        self.augment_prob = augment_prob

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer '{tokenizer_name}': {e}")

        # Load dataset
        try:
            if excel_path.endswith(".csv"):
                self.df = pd.read_csv(excel_path, encoding="ISO-8859-9")
            elif excel_path.endswith(".xlsx") or excel_path.endswith(".xls"):
                self.df = pd.read_excel(excel_path, engine="openpyxl")
            else:
                # Try to detect automatically
                if "csv" in excel_path.lower():
                    self.df = pd.read_csv(excel_path, encoding="ISO-8859-9")
                else:
                    self.df = pd.read_excel(excel_path, engine="openpyxl")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load file '{excel_path}': {e}")

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
                return librosa.get_duration(path=path)
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

    def normalize_audio_length_dynamic(
            self,
            audio: torch.Tensor,
            sample_rate: int = 16000,
            target_seconds: float = 7.0,
            short_threshold: float = 2.0
    ) -> torch.Tensor:
        """
        Normalize the length of an audio tensor to a target duration.
        Short audios are repeated and zero-padded; longer audios are either padded or repeated randomly.

        Args:
            audio (torch.Tensor): Audio tensor of shape (1, n_samples) or (channels, n_samples)
            sample_rate (int): Audio sample rate (e.g., 16000)
            target_seconds (float): Desired target length in seconds
            short_threshold (float): Threshold below which audio is considered very short

        Returns:
            torch.Tensor: Audio tensor normalized to target length
        """
        # Import random here to avoid potential circular import issues
        import random

        # Convert sample_rate to scalar if it's a tensor
        if isinstance(sample_rate, torch.Tensor):
            sample_rate = sample_rate.item()

        # Convert other parameters to scalars if they're tensors
        if isinstance(target_seconds, torch.Tensor):
            target_seconds = target_seconds.item()
        if isinstance(short_threshold, torch.Tensor):
            short_threshold = short_threshold.item()

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
            zero_padding = torch.zeros(audio.shape[0], remaining_samples, dtype=audio.dtype, device=audio.device)
            audio_normalized = torch.cat([audio_repeated, zero_padding], dim=1)

        else:
            # Random choice: either pad with zeros or repeat
            if random.random() < 0.5:
                # Zero-pad or truncate
                pad_samples = target_samples - n_samples
                if pad_samples > 0:
                    zero_padding = torch.zeros(audio.shape[0], pad_samples, dtype=audio.dtype, device=audio.device)
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
        row = self.df.iloc[idx]

    # --- Load audio ---
        audio_np, sr = librosa.load(row["Audio"], sr=self.sample_rate)
        audio_tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)

    # --- Apply augmentations if enabled ---
        if self.augment and audio_tensor is not None:
            aug_choice = random.choices(
                ["time_stretch", "pitch", "noise", "none"],
                weights=[0.3, 0.3, 0.3, 0.1],
                k=1
            )[0]

            if aug_choice == "time_stretch":
              audio_tensor = time_stretch_audio(audio_tensor, min_rate=0.9, max_rate=1.1)
            elif aug_choice == "pitch":
              audio_np = librosa.effects.pitch_shift(audio_np, sr=sr, n_steps=random.uniform(-1, 1))
              audio_tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
            elif aug_choice == "noise":
              noise = np.random.normal(0, 0.001, audio_np.shape)
              audio_np = audio_np + noise
              audio_tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        # 'none' does nothing

    # --- Normalize length ---
        audio_tensor = self.normalize_audio_length_dynamic(audio_tensor, sample_rate=self.sample_rate)

    # --- Tokenize transcript ---
        transcript = str(row["Transcript"]).strip()
        if self.augment and random.random() < 0.5:
          transcript = synonym_replacement(transcript)

        encoded = self.tokenizer(
          transcript,
          max_length=self.max_length,
          padding="max_length",
          truncation=True,
          return_tensors="pt"
    )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

    # --- Label ---
        label_id = self.label_map[row["Label"]]

        return {
        "audio": audio_tensor,
        "labels": torch.tensor(label_id, dtype=torch.long),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "transcript": transcript
    }



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