import json
import yaml
import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils import split_dirichlet


if __name__ == "__main__":
    # === load config ===
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    # === load Dolly original json ===
    dataset_path = "dolly15k/databricks-dolly-15k.json"
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # === normalize format ===
    def normalize_dolly(example):
        return {
            "instruction": example.get("instruction", ""),
            "input": example.get("context", ""),
            "output": example.get("response", ""),
            "category": example.get("category", example.get("instruction", ""))
        }

    data = [normalize_dolly(d) for d in data]

    # === split train/test ===
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    # === Dirichlet split for train & test with same distribution ===
    client_num = config["client_num"]
    alpha = 0.5

    train_props = split_dirichlet(train_data, "dolly15k/train", client_num, alpha)
    split_dirichlet(test_data, "dolly15k/test", client_num, alpha, fixed_props=train_props)
