import copy
import torch
from peft import LoraConfig
from transformers import Trainer
from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record
from utils.model_utils import load_model
from collections import defaultdict


# 客户端类
class Client(FTBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.local_rank = args.lora_rank  # 客户端本地的LoRA秩

    @time_record
    def run(self, model):
        print(f"Client {self.id} start local training with LoRA rank {self.local_rank}")

        # 获取客户端的模型副本
        client_model = copy.deepcopy(model)
        client_model.train()

        # 设置训练参数
        trainable = []
        for name, param in client_model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True  # 启用LoRA参数的梯度
                trainable.append(name)
            else:
                param.requires_grad = False  # 非LoRA参数冻结

        # 训练数据预处理
        from transformers import DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=client_model,
            padding=True,
            return_tensors="pt"
        )

        # 开始训练
        trainer = Trainer(
            model=client_model,
            args=self.training_args,
            train_dataset=self.dataset['train'],
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )

        trainer.train()

        # 保存LoRA权重
        self.lora = {k: v for k, v in client_model.state_dict().items() if "lora_" in k}
        print(f"Client {self.id} finished training.")

    def model2tensor(self):
        """提取LoRA参数作为张量"""
        state_dict = self.model.state_dict()
        lora_params = {k: v for k, v in state_dict.items() if "lora_" in k}
        return lora_params

    def tensor2model(self, tensor_dict):
        """将张量参数加载到模型"""
        current_state = self.model.state_dict()
        current_state.update(tensor_dict)
        self.model.load_state_dict(current_state)


# 服务器端类
class Server(FTBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        self.model, _ = load_model(args)
        self.round = 0
        self.client_models = []

    def run(self):
        """服务器运行逻辑：采样、训练和聚合"""
        self.sample()
        self.local_run()
        self.aggregate()

    def local_run(self):
        """在服务器上启动所有客户端的训练"""
        for client in self.clients:
            client.run(self.model)

    def aggregate(self):
        """LoRA参数的聚合：权重更新和分配"""
        data_sum = sum([len(client.dataset['train']) for client in self.clients])
        aggregated = defaultdict(lambda: 0)

        for client in self.clients:
            client_lora = client.lora  # 获取客户端的LoRA参数
            for k, v in client_lora.items():
                aggregated[k] += v * len(client.dataset['train']) / data_sum

        # 将聚合后的LoRA权重加载到全局模型中
        self.model.load_state_dict(aggregated, strict=False)
        print("Aggregated LoRA weights.")

    def test_all(self):
        """测试所有客户端的模型"""
        all_metrics = []
        for client in self.clients:
            print(f"Testing on client {client.id} ...")
            metrics = client.local_test(self.model)
            all_metrics.append(metrics)

        avg_loss = sum(m["eval_loss"] for m in all_metrics) / len(all_metrics)
        avg_perplexity = sum(m["perplexity"] for m in all_metrics) / len(all_metrics)
        return {'loss': avg_loss, 'perplexity': avg_perplexity}
