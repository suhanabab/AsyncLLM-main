from alg.asyncbase import AsyncBaseClient, AsyncBaseServer, Status
from alg.ftbase import FTBaseClient
from utils.time_utils import time_record
import copy
import torch


class Client(FTBaseClient):
    def __init__(self, id, args):
        FTBaseClient.__init__(self, id, args)
        self.status = Status.IDLE
        self.training_time = 1.0 
        self.id = id
        self.args = args
        # self.model = None  # 确保 model 属性存在

    @time_record
    def run(self):
        if getattr(self, 'model', None) is None:
            from utils.model_utils import load_model
            self.model, _ = load_model(self.args)

        device = next(self.model.parameters()).device
        print(f"Client {self.id} training on device: {device}")

        # 关闭 pin_memory
        original_pin_memory = self.training_args.dataloader_pin_memory
        self.training_args.dataloader_pin_memory = False


        trainable = []
        for name, param in self.model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
                trainable.append(name)
            else:
                param.requires_grad = False
        # print("Trainable LoRA params:", trainable)

        # ---- debug: 检查梯度是否真的开启 ----
        # print("\n=== Checking LoRA requires_grad ===")
        # for n, p in self.model.named_parameters():
        #     if "lora" in n.lower():
        #         print(n, p.requires_grad)
        # print("===================================\n")

        # tokenizer + dataset 预处理
        def preprocess(batch):
            text = batch["text"]
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

        # 正确的数据整理器
        from transformers import DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )

        from transformers import Trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset['train'],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        self.lora = {k: v.detach().clone() for k, v in self.model.state_dict().items() if "lora_" in k}

        # 恢复 pin_memory
        self.training_args.dataloader_pin_memory = original_pin_memory

        del trainer
        del self.model
        self.model = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def clone_model(self, server):
        self.model = copy.deepcopy(server.model)

    def model2tensor(self):
        if hasattr(self, 'lora') and self.lora:
            return self.lora

        if getattr(self, 'model', None) is None:
            print(f"Warning: Client {self.id} model is None in model2tensor and no saved lora")
            return {}

        state_dict = self.model.state_dict()
        lora_params = {k: v for k, v in state_dict.items() if "lora_" in k}
        return lora_params

    def tensor2model(self, tensor_dict):
        if self.model is None:
            print(f"Warning: Client {self.id} model is None in tensor2model")
            return
        current_state = self.model.state_dict()
        current_state.update(tensor_dict)
        self.model.load_state_dict(current_state)

    def local_test(self, model=None):
        if model is None:
            model = self.model
        if model is None:
            from utils.model_utils import load_model
            model, _ = load_model(self.args)
        return super().local_test(model)


class Server(AsyncBaseServer):
    def __init__(self, args, clients):
        # 初始化基础模型
        from utils.model_utils import load_model
        self.model, _ = load_model(args)
        super().__init__(args, clients)

        self.round = 0
        for client in self.clients:
            client.model = copy.deepcopy(self.model)


    def local_run(self):
        pass

    def run(self):
        self.round += 1

        # 1. 采样客户端
        sampled_clients = self.sample()
        if not sampled_clients:
            if self.uplink():
                self.aggregate()
                self.update_status()
            return

        # 2. 下发模型
        self.downlink(sampled_clients)

        # 3. 客户端训练
        self.client_update(sampled_clients)

        # 4. 处理完成的客户端
        if self.uplink():
            self.aggregate()
            self.update_status()

    def test_all(self):
        all_metrics = []
        for client in self.clients:
            original_state = {}
            if client.model is not None:
                original_state = client.model2tensor()

           
            client.clone_model(self)
            metrics = client.local_test(client.model)
            all_metrics.append(metrics)

            
            if original_state:
                client.tensor2model(original_state)

        avg_loss = sum(m["eval_loss"] for m in all_metrics) / len(all_metrics)
        avg_perplexity = sum(m["perplexity"] for m in all_metrics) / len(all_metrics)
        return {'loss': avg_loss, 'perplexity': avg_perplexity}


def add_args(parser):

    return parser
