import numpy as np

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record
import copy
from collections import defaultdict
from transformers import Trainer  # 确保导入Trainer（若ftbase已导入可省略）


class Client(FTBaseClient):  # 类名必须为Client，适配main.py动态导入
    def __init__(self, id, args):
        super().__init__(id, args)
        # 1. 初始化FedSA-LoRA核心参数
        self.local_lora_B = {}  # 客户端本地B矩阵（不上传）
        self.global_lora_A = {}  # 服务端广播的全局A矩阵
        self.lora_variant = args.lora_variant  # LoRA变体（lora/rslora/vera）
        self.lora_rank = args.lora_rank  # LoRA秩（论文：lora=8，vera=256）
        self.lora_scaling = args.lora_scaling  # LoRA缩放因子（论文：lora=16）

        # 2. 适配VeRA的特殊初始化（若使用VeRA）
        if self.lora_variant == "vera":
            self.vera_A_d_init = 0.1  # 论文中VeRA的A_d初始值
            self.vera_B_b_init = 0  # 论文中VeRA的B_b初始值

    @time_record
    def run(self, model):
        """重写run方法：合并全局A+本地B训练，仅上传A矩阵"""
        # 步骤1：合并全局A矩阵到模型（首次训练用初始化A，后续用服务端广播的A）
        if self.global_lora_A:
            model.load_state_dict(self.global_lora_A, strict=False)

        # 步骤2：深拷贝模型并执行本地训练（同时更新A/B矩阵）
        client_model = copy.deepcopy(model)
        client_model.train()

        # 复用ftbase的TrainingArguments配置，无需修改
        Trainer(
            model=client_model,
            args=self.training_args,
            train_dataset=self.dataset['train'],
            processing_class=self.tokenizer,
        ).train()

        # 步骤3：分离A/B矩阵，仅保存A矩阵到self.lora（供服务端聚合）
        self._extract_lora_matrices(client_model)

    def _extract_lora_matrices(self, model):
        """根据LoRA变体分离A/B矩阵，适配参数命名规则"""
        lora_state = model.state_dict()
        if self.lora_variant in ["lora", "rslora"]:
            # LoRA/rsLoRA：参数命名含"lora_A"/"lora_B"（需与你的LoRA实现一致）
            self.lora = {k: v for k, v in lora_state.items() if "lora_A" in k}
            self.local_lora_B = {k: v for k, v in lora_state.items() if "lora_B" in k}
        elif self.lora_variant == "vera":
            # VeRA：参数命名含"lora_A_d"/"lora_B_b"（对应论文的A_d/B_b缩放向量）
            self.lora = {k: v for k, v in lora_state.items() if "lora_A_d" in k}
            self.local_lora_B = {k: v for k, v in lora_state.items() if "lora_B_b" in k}

    def load_global_A(self, global_lora_A):
        """接收服务端广播的全局A矩阵（供下一轮训练使用）"""
        self.global_lora_A = copy.deepcopy(global_lora_A)

    def merge_lora(self, model):
        """合并全局A矩阵和本地B矩阵到模型，用于训练/测试"""
        merged_state = copy.deepcopy(model.state_dict())
        merged_state.update(self.global_lora_A)  # 覆盖全局A
        merged_state.update(self.local_lora_B)  # 覆盖本地B
        model.load_state_dict(merged_state, strict=False)
        return model

    def local_test(self, model):
        """重写测试：使用全局A+本地B的合并模型"""
        test_model = self.merge_lora(model)  # 关键：测试前合并A/B
        return super().local_test(test_model)  # 复用ftbase的测试逻辑


class Server(FTBaseServer):  # 类名必须为Server，适配main.py动态导入
    def __init__(self, args, clients):
        super().__init__(args, clients)
        # 1. 初始化FedSA-LoRA服务端参数
        self.global_lora_A = {}  # 聚合后的全局A矩阵
        self.lora_variant = args.lora_variant
        self.round = 0  # 与main.py的round计数对齐

    def run(self):
        """重写run方法：采样→本地训练→聚合A→广播A"""
        self.sample()  # 复用ftbase的采样逻辑（可在ftbase中补充客户端采样）
        self.local_run()  # 客户端本地训练（合并A/B）
        self.aggregate()  # 仅聚合A矩阵
        self.broadcast_A()  # 广播全局A矩阵给所有客户端

    def sample(self):
        """补充客户端采样逻辑（若需部分客户端参与，适配args.sr采样率）"""
        if self.args.sr < 1.0:  # args.sr为采样率（config.yaml中已配置）
            sample_num = int(len(self.clients) * self.args.sr)
            self.selected_clients = np.random.choice(self.clients, sample_num, replace=False)
        else:
            self.selected_clients = self.clients  # 全量参与

    def local_run(self):
        """客户端本地训练：为选中的客户端分配全局A矩阵并执行训练"""
        for client in self.selected_clients:
            # 为客户端分配当前全局A矩阵对应的模型
            client_model = client.merge_lora(copy.deepcopy(self.model))
            client.run(client_model)  # 调用客户端的run方法

    def aggregate(self):
        """仅聚合选中客户端的A矩阵（加权平均，权重为客户端数据量）"""
        # 计算每个客户端的权重（数据量占比）
        total_data = sum([len(client.dataset['train']) for client in self.selected_clients])
        aggregated_A = defaultdict(lambda: 0.0)

        # 加权聚合A矩阵
        for client in self.selected_clients:
            client_A = client.lora
            weight = len(client.dataset['train']) / total_data
            for k, v in client_A.items():
                aggregated_A[k] += v * weight  # 加权累加

        # 更新全局A矩阵
        self.global_lora_A = aggregated_A
        print(f"Round {self.round}: Aggregated global LoRA-A matrices (variant: {self.lora_variant})")

    def broadcast_A(self):
        """将全局A矩阵广播给所有客户端（无论是否被选中）"""
        for client in self.clients:
            client.load_global_A(self.global_lora_A)

    def test_all(self):
        """复用ftbase的测试逻辑，客户端会自动合并A/B"""
        return super().test_all()