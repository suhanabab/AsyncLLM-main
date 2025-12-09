import os
import math
import json
import hashlib
import random
import numpy as np
from collections import defaultdict


def split_uniform(data, dir_path, client_num):
    os.makedirs(dir_path, exist_ok=True)
    size = len(data)
    chunk_size = math.ceil(size / client_num)

    for i in range(client_num):
        start = i * chunk_size
        end = min(start + chunk_size, size)
        chunk = data[start:end]
        if not chunk: break

        out_file = f"{dir_path}/{i}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            for item in chunk:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_dirichlet(data, dir_path, client_num, alpha, fixed_props=None):
    os.makedirs(dir_path, exist_ok=True)

    # ---- 用 category 字段做 label ----
    cats = [d.get("category", "") for d in data]
    uniq = sorted(list(set(cats)))
    cat2id = {c: i for i, c in enumerate(uniq)}
    labels = np.array([cat2id[c] for c in cats])

    # 每个 label 的样本索引
    label_set = sorted(set(labels))
    idx_by_label = {l: np.where(labels == l)[0] for l in label_set}

    # 生成或复用 Dirichlet 分布
    if fixed_props is None:
        fixed_props = {
            l: np.random.dirichlet([alpha] * client_num)
            for l in label_set
        }

    # 分配结果
    client_indices = [[] for _ in range(client_num)]

    for l, idxs in idx_by_label.items():
        proportions = fixed_props[l]
        split_sizes = (proportions * len(idxs)).astype(int)

        start = 0
        for i in range(client_num):
            end = start + split_sizes[i]
            part = idxs[start:end]
            client_indices[i].extend(part.tolist())
            start = end

    # 写文件
    for i in range(client_num):
        out_file = f"{dir_path}/{i}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            for idx in client_indices[i]:
                f.write(json.dumps(data[idx], ensure_ascii=False) + "\n")

    return fixed_props


def split_threeway(data, dir_path):
    os.makedirs(dir_path, exist_ok=True)

    size = len(data) // 3
    parts = [data[:size], data[size:2*size], data[2*size:]]

    for i in range(3):
        with open(f"{dir_path}/{i}.json", "w", encoding="utf-8") as f:
            for item in parts[i]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_fedit(data, dir_path, client_num):
    os.makedirs(dir_path, exist_ok=True)

    import hashlib

    def get_inst(d):

        if "instruction" in d:
            return d["instruction"]


        if "conversations" in d and len(d["conversations"]) > 0:
            first = d["conversations"][0]
            if first.get("from") == "human":
                return first.get("value", "")
            return first.get("value", "")


        return ""

    def hash_inst(inst):
        return int(hashlib.md5(inst.encode()).hexdigest(), 16) % client_num

    buckets = [[] for _ in range(client_num)]

    for d in data:
        inst = get_inst(d)
        idx = hash_inst(inst)
        buckets[idx].append(d)


    for i in range(client_num):
        with open(f"{dir_path}/{i}.json", "w", encoding="utf-8") as f:
            for item in buckets[i]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_dataset(cfg, dataset):
    client_num = cfg['client_num']
    dir_path = cfg['dir_path']
    os.makedirs(dir_path, exist_ok=True)
    split_type = cfg['split']

    if split_type == 'uniform':
        split_uniform(dataset['train'], f'{dir_path}/train', client_num)
        split_uniform(dataset['test'], f'{dir_path}/test', client_num)

    elif split_type == 'dirichlet':
        alpha = cfg.get('alpha', 0.5)
        split_dirichlet(dataset['train'], f'{dir_path}/train', client_num, alpha)
        split_dirichlet(dataset['test'], f'{dir_path}/test', client_num, alpha)

    # elif split_type == 'threeway':
    #     split_threeway(dataset['train'], f'{dir_path}/train')
    #     split_threeway(dataset['test'], f'{dir_path}/test')

    elif split_type == 'fedit':
        split_fedit(dataset['train'], f'{dir_path}/train', client_num)
        split_fedit(dataset['test'], f'{dir_path}/test', client_num)