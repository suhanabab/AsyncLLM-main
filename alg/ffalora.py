# alg/ffalora.py
import copy
import torch
from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record

class Client(FTBaseClient):
    """
    FFA-LoRA 客户端实现（只训练 lora_B，固定 lora_A 与 base model）
    继承 FTBaseClient 的数据加载、tokenizer、training_args 等。
    """
    @time_record
    def run(self, model):
        """model: 从 server 传入的 peft 模型（带 LoRA adapter 的模型）"""
        print(f"\n----- FFA Client {self.id} start local training -----")

        # 复制一份模型到本地进行训练（避免改变 server.model）
        client_model = copy.deepcopy(model)
        client_model.train()

        # ---- 1) 冻结基础模型参数 ----
        for name, p in client_model.named_parameters():
            # 先把所有参数默认冻结
            p.requires_grad = False

        # ---- 2) 冻结 lora_A，解冻 lora_B ----
        lora_B_names = []
        lora_A_names = []
        for name, p in client_model.named_parameters():
            n_lower = name.lower()
            if "lora_b" in n_lower:               # 仅训练 B
                p.requires_grad = True
                lora_B_names.append(name)
            elif "lora_a" in n_lower:             # 保持 A 冻结
                p.requires_grad = False
                lora_A_names.append(name)

        print(f"[FFA Client {self.id}] trainable lora_B params: {len(lora_B_names)}; frozen lora_A params: {len(lora_A_names)}")

        # ---- 3) 确保数据格式（与原 FTBaseClient 一致）----
        # 如果需要对 dataset 做 preprocess（string -> token dict），复用 FTBaseClient 的方法
        def preprocess(batch):
            text = batch.get("text") or batch.get("input_text") or batch.get("question") or ""
            tokens = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=2048
            )
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens

        if "input_ids" not in self.dataset["train"].column_names:
            self.dataset["train"] = self.dataset["train"].map(preprocess)

        # ---- 4) 使用 Trainer 进行训练（Trainer 会自动只更新 requires_grad=True 的参数）----
        from transformers import Trainer, DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=client_model,
            padding=True,
            return_tensors="pt"
        )

        trainer = Trainer(
            model=client_model,
            args=self.training_args,
            train_dataset=self.dataset['train'],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        trainer.train()

        # ---- 5) 训练完成：只保存 lora_B 到 self.lora（以便上行/聚合） ----
        # 将参数移到 cpu 并 clone，避免占用 GPU 显存或被后续 del 影响
        self.lora = {k: v.detach().cpu().clone() for k, v in client_model.state_dict().items() if "lora_b" in k.lower()}
        print(f"[FFA Client {self.id}] saved {len(self.lora)} lora_B params for upload")

        # 记录训练时间/通信时间（如果你在 FTBaseClient 中已有 delay/bandwidth 逻辑，保留）
        # 这里保持与 FTBaseClient.run 相同的时间统计逻辑（如果需要，将 base_train_time 等放回）

        # ---- 6) 释放显存 ----
        del trainer
        del client_model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print(f"----- FFA Client {self.id} finished -----\n")


class Server(FTBaseServer):
    """
    服务器端行为与 FTBaseServer 一样：local_run -> aggregate
    由于客户端上传的 self.lora 只包含 lora_B，aggregate 也只需融合这些 B 参数。
    这里直接复用 FTBaseServer 的聚合实现（假定 FTBaseServer.aggregate 使用 client.lora）。
    """
    def run(self):
        # 保持和 fedit/FTBaseServer 一致的流程
        self.sample()
        self.local_run()
        self.aggregate()
