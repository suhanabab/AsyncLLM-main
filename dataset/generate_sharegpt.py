import os
import json
import yaml
from sklearn.model_selection import train_test_split

from utils import split_dataset

if __name__ == "__main__":
    # === load config ===
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    # === load local ShareGPT dataset ===
    dataset_path = "sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)   # list

    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    split_dataset(config, {"train": train_data, "test": test_data})


