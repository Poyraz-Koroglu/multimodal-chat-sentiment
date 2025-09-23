import os
import pandas as pd

ROOT_DATASETS_DIR = "C:/Users/poyraz.koroglu/Desktop/Trial-Dataset-root"

DATASET_LABEL_MAPS = {
    "dataset1": {"negative": 1},
    "dataset2": {"neutral": 0},
    # Add more mappings if needed
}

AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3"]

all_rows = []

for dataset_name in os.listdir(ROOT_DATASETS_DIR):
    dataset_path = os.path.join(ROOT_DATASETS_DIR, dataset_name)
    if not os.path.isdir(dataset_path):
        continue

    print(f"Processing dataset: {dataset_name}")
    label_map = DATASET_LABEL_MAPS.get(dataset_name, None)

    for label_name in os.listdir(dataset_path):
        label_folder = os.path.join(dataset_path, label_name)
        if not os.path.isdir(label_folder):
            continue

        label_id = label_map[label_name] if label_map else label_name

        # Load transcripts if available
        transcript_path = os.path.join(label_folder, "transcript.txt")
        transcripts = []
        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcripts = [line.strip() for line in f.readlines()]

        # List and sort audio files by filename
        audio_files = sorted(
            [f for f in os.listdir(label_folder) if any(f.lower().endswith(ext) for ext in AUDIO_EXTENSIONS)]
        )

        for idx, audio_file in enumerate(audio_files):
            audio_path = os.path.join(label_folder, audio_file)
            # Match transcript line if exists
            transcript_text = transcripts[idx] if idx < len(transcripts) else ""
            all_rows.append([audio_path, label_id, transcript_text])

df_all = pd.DataFrame(all_rows, columns=["audio", "label", "transcript"])

output_file = os.path.join(ROOT_DATASETS_DIR, "metadata.csv")
df_all.to_csv(output_file, index=False)
print(f"Unified metadata.csv created with {len(df_all)} samples at {output_file}")
