import os
import pandas as pd



ROOT_DATASETS_DIR = "C:/Users/poyraz.koroglu/Desktop/datasets-root"

DATASET_LABEL_MAPS = {
    "dataset1": {"neutral": 0, "negative": 1},
    "dataset2": {"neutral": 0, "negative": 1},
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

        for file in os.listdir(label_folder):
            if any(file.lower().endswith(ext) for ext in AUDIO_EXTENSIONS):
                audio_path = os.path.join(label_folder, file)
                all_rows.append([audio_path, label_id])

df_all = pd.DataFrame(all_rows, columns=["audio", "label"])

output_file = os.path.join(ROOT_DATASETS_DIR, "metadata.csv")
df_all.to_csv(output_file, index=False)
print(f"Unified metadata.csv created with {len(df_all)} samples at {output_file}")
