# alg/ffalora.py
import copy
import torch
from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record

class Client(FTBaseClient):
   
    @time_record
    def run(self, model):
      
        print(f"\n----- FFA Client {self.id} start local training -----")

        client_model = copy.deepcopy(model)
        client_model.train()

       
        for name, p in client_model.named_parameters():
            p.requires_grad = False

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

        self.lora = {k: v.detach().cpu().clone() for k, v in client_model.state_dict().items() if "lora_b" in k.lower()}
        print(f"[FFA Client {self.id}] saved {len(self.lora)} lora_B params for upload")


        del trainer
        del client_model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print(f"----- FFA Client {self.id} finished -----\n")


class Server(FTBaseServer):
    
    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()
