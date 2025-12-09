import numpy as np

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record
import copy
from collections import defaultdict
from transformers import Trainer  


class Client(FTBaseClient):  
    def __init__(self, id, args):
        super().__init__(id, args)
        self.local_lora_B = {}  # 客户端本地B矩阵
        self.global_lora_A = {}  # 服务端广播的全局A矩阵
        self.lora_variant = args.lora_variant  # LoRA变体
        self.lora_rank = args.lora_rank  # LoRA秩（论文：lora=8，vera=256）
        self.lora_scaling = args.lora_scaling  # LoRA缩放因子（论文：lora=16）

        # 2. 适配VeRA的特殊初始化
        if self.lora_variant == "vera":
            self.vera_A_d_init = 0.1  # 论文中VeRA的A_d初始值
            self.vera_B_b_init = 0  # 论文中VeRA的B_b初始值

    @time_record
    def run(self, model):
        # 合并全局A矩阵到模型，首次训练用初始化A，后续用服务端广播的A
        if self.global_lora_A:
            model.load_state_dict(self.global_lora_A, strict=False)

        # 深拷贝模型并执行本地训练，同时更新A/B
        client_model = copy.deepcopy(model)
        client_model.train()

        Trainer(
            model=client_model,
            args=self.training_args,
            train_dataset=self.dataset['train'],
            processing_class=self.tokenizer,
        ).train()

        self._extract_lora_matrices(client_model)

    def _extract_lora_matrices(self, model):
        lora_state = model.state_dict()
        if self.lora_variant in ["lora", "rslora"]:

            self.lora = {k: v for k, v in lora_state.items() if "lora_A" in k}
            self.local_lora_B = {k: v for k, v in lora_state.items() if "lora_B" in k}
        elif self.lora_variant == "vera":

            self.lora = {k: v for k, v in lora_state.items() if "lora_A_d" in k}
            self.local_lora_B = {k: v for k, v in lora_state.items() if "lora_B_b" in k}

    def load_global_A(self, global_lora_A):

        self.global_lora_A = copy.deepcopy(global_lora_A)

    def merge_lora(self, model):

        merged_state = copy.deepcopy(model.state_dict())
        merged_state.update(self.global_lora_A)  # 覆盖全局A
        merged_state.update(self.local_lora_B)  # 覆盖本地B
        model.load_state_dict(merged_state, strict=False)
        return model

    def local_test(self, model):
        test_model = self.merge_lora(model) 
        return super().local_test(test_model) 


class Server(FTBaseServer):  
    def __init__(self, args, clients):
        super().__init__(args, clients)
       
        self.global_lora_A = {}  
        self.lora_variant = args.lora_variant
        self.round = 0  # 与main.py的round计数对齐

    def run(self):
       
        self.sample()  
        self.local_run()  
        self.aggregate()  
        self.broadcast_A()  

    def sample(self):
        if self.args.sr < 1.0:  # args.sr为采样率
            sample_num = int(len(self.clients) * self.args.sr)
            self.selected_clients = np.random.choice(self.clients, sample_num, replace=False)
        else:
            self.selected_clients = self.clients  

    def local_run(self):
      
        for client in self.selected_clients:
           
            client_model = client.merge_lora(copy.deepcopy(self.model))
            client.run(client_model)  # 调用客户端的run方法

    def aggregate(self):
       
        total_data = sum([len(client.dataset['train']) for client in self.selected_clients])
        aggregated_A = defaultdict(lambda: 0.0)

        for client in self.selected_clients:
            client_A = client.lora
            weight = len(client.dataset['train']) / total_data
            for k, v in client_A.items():
                aggregated_A[k] += v * weight  # 加权累加

        self.global_lora_A = aggregated_A
        print(f"Round {self.round}: Aggregated global LoRA-A matrices (variant: {self.lora_variant})")

    def broadcast_A(self):

        for client in self.clients:
            client.load_global_A(self.global_lora_A)

    def test_all(self):

        return super().test_all()
