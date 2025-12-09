import json
import yaml
import os

from sklearn.model_selection import train_test_split

from utils import split_dataset

if __name__ == "__main__":
    # === Load config ===
    with open("config.yaml", "r") as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    # === Load wizard dataset ===
    dataset_path = os.path.join("wizard", "WizardLM_evol_instruct_V2_143k.json")

    with open(dataset_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        try:
            # JSON array
            data = json.loads(content)
        except json.JSONDecodeError:
            # JSONL fallback
            data = [json.loads(line) for line in content.splitlines() if line.strip()]

    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    split_dataset(config, {"train": train_data, "test": test_data})
