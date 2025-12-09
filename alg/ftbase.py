import copy
import os
import math

from transformers import TrainingArguments, Trainer
from alg.base import BaseClient, BaseServer
from datasets import load_dataset
from utils.model_utils import load_model
import time
from peft import LoraConfig

from transformers import AutoTokenizer

class FTBaseClient(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.training_args = TrainingArguments(
            output_dir=f"./client{self.id}",  # where to save the output log
            per_device_train_batch_size=args.bs,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=args.epoch,
            logging_steps=10,  # gap of steps between two logging
            save_steps=50000,
            save_total_limit=2,
            fp16=True,
            optim="adamw_torch"
        )

        # ffalora的配置
        self.lora_config = LoraConfig(
            r=args.lora_rank,  # 论文中r=8（从config.yaml的lora_rank读取）
            lora_alpha=args.lora_scaling,  # 论文中α=8（FFA-LoRA可忽略，仅为兼容）
            target_modules=["q_proj", "v_proj"],  # 目标层（根据模型调整，如Qwen/BERT的注意力层）
            lora_dropout=0.05,  # 论文中默认值
            bias="none",
            task_type="CAUSAL_LM"  # 任务类型（GSM-8K为因果语言建模）
        )

        self.load_data()
        self.lora = {}

    def load_data(self):
        train_dir = os.path.join('./dataset', self.args.dataset, f'train/{self.id}.json')
        test_dir = os.path.join('./dataset', self.args.dataset, f'test/{self.id}.json')

        # self.dataset = load_dataset("json", data_files={'train': train_dir, 'test': test_dir})
        self.dataset = load_dataset(
            "json",
            data_files={'train': train_dir, 'test': test_dir},
            cache_dir=None,  # 不使用缓存目录
            keep_in_memory=True,  # 全部读入内存
            download_mode="force_redownload"  # 强制重新加载
        )
        self.dataset['train'] = self.dataset['train'].map(self.format_example)
        self.dataset['test'] = self.dataset['test'].map(self.format_example)

    def format_example(self, example):
        prompt = f"Instruct: {example['question']}\nAnswer:"
        return {
            "input_ids": self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512).input_ids[0],
            "labels": self.tokenizer(example["answer"], return_tensors="pt", truncation=True, padding="max_length", max_length=512).input_ids[0]
        }

    def run(self, model):
        print(f"\n----- Client {self.id} start local training -----")

        # ====== 开始计时 ======
        start_time = time.time()

        client_model = copy.deepcopy(model)
        client_model.train()

        Trainer(
            model=client_model,
            args=self.training_args,
            train_dataset=self.dataset['train'],
            processing_class=self.tokenizer,
        ).train()

        # ====== 结束计时 ======
        end_time = time.time()
        base_train_time = end_time - start_time

        # 应用慢速设备延迟
        time_after_delay = base_train_time * self.delay

        # 上传 LoRA 参数时的通信时间
        upload_bytes = sum(p.numel() * p.element_size() for p in client_model.parameters())
        comm_time = upload_bytes * 8 / (1024 * 1024) / self.bandwidth  # MB/s

        # 总时间
        self.training_time = time_after_delay + comm_time * 2

        print(f"Client {self.id}:")
        print(f"  - Base training time:     {base_train_time:.4f} s")
        print(f"  - Delay ×{self.delay:.2f}:       {time_after_delay:.4f} s")
        print(f"  - Comm time (upload+down): {comm_time * 2:.4f} s")
        print(f"  --> Total training time:   {self.training_time:.4f} s")

        # 保存 LoRA 参数
        self.lora = {k: v for k, v in client_model.state_dict().items() if "lora_" in k}

        print(f"----- Client {self.id} finished -----\n")

    def local_run(self):
        print("\n========== Local Execution Start ==========")
        times = []

        for client in self.clients:
            client.run(self.model)
            times.append(client.training_time)

        self.round += 1
        self.wall_clock_time = max(times)

        print(f"Round {self.round}: wall-clock time = {self.wall_clock_time:.4f} s")
        print("===========================================\n")

    def local_test(self, model):
        model.eval()

        trainer = Trainer(
            model=model,
            args=self.training_args,
            eval_dataset=self.dataset["test"],
            processing_class=self.tokenizer
        )
        metrics = trainer.evaluate()
        if "eval_loss" in metrics and metrics["eval_loss"] is not None and not math.isnan(metrics["eval_loss"]):
            metrics["perplexity"] = float(math.exp(metrics["eval_loss"]))
        else:
            metrics["perplexity"] = float("inf")

        print(f"Client {self.id} local test metrics:", metrics)
        return metrics

class FTBaseServer(BaseServer):
    def __init__(self, args, clients):
        BaseServer.__init__(self, args, clients)
        self.model, _ = load_model(args)
        self.client_models = []

        self.round = 0

    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()

    def sample(self):
        pass

    def local_run(self):
        for client in self.clients: client.run(self.model)

    def aggregate(self):
        data_sum = sum([len(client.dataset['train']) for client in self.clients])
        from collections import defaultdict
        aggregated = defaultdict(lambda: 0)

        for client in self.clients:
            model = client.lora
            for k, v in model.items():
                aggregated[k] = aggregated[k] + v * len(client.dataset['train']) / data_sum

        self.model.load_state_dict(aggregated, strict=False)
        print("Aggregated model updated.")

    def test_all(self):
        all_metrics = []
        for client in self.clients:
            print(f"Testing on client {client.id} ...")
            metrics = client.local_test(self.model)
            all_metrics.append(metrics)


        avg_loss = sum(m["eval_loss"] for m in all_metrics) / len(all_metrics)
        avg_perplexity = sum(m["perplexity"] for m in all_metrics) / len(all_metrics)
        return {'loss': avg_loss, 'perplexity': avg_perplexity}