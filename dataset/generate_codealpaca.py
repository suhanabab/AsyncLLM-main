import json
import yaml

from utils import split_dataset
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # === load config ===
    with open("config.yaml", "r") as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    # === load raw CodeAlpaca json ===
    with open("codealpaca/code_alpaca_20k.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    split_dataset(config, {"train": train_data, "test": test_data})
