import heapq
import random
import numpy as np
import copy
from abc import ABC, abstractmethod

import torch

from alg.base import BaseClient, BaseServer
from enum import Enum
import math



class Status(Enum):
    IDLE = 1
    ACTIVE = 2


class AsyncBaseClient(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.status = Status.IDLE
        self.training_time = random.uniform(1.0, 5.0)  
        self.model = None  

    def clone_model(self, server):
        self.model = copy.deepcopy(server.model)
        device = server.args.device if hasattr(server.args, 'device') else "cuda:0"
        if isinstance(device, int):
            device = f"cuda:{device}"
        elif isinstance(device, str) and device.startswith('cuda') and ":" not in device:
            device = f"{device}:0"
        self.model.to(device)

    def model2tensor(self):
        if self.model is None:
            return {}
        state_dict = self.model.state_dict()
        lora_params = {k: v for k, v in state_dict.items() if "lora_" in k}
        return lora_params

    def tensor2model(self, tensor_dict):
        if self.model is None:
            return
        current_state = self.model.state_dict()
        current_state.update(tensor_dict)
        self.model.load_state_dict(current_state)

    def run(self):
        raise NotImplementedError


class AsyncBaseServer(BaseServer, ABC):
    def __init__(self, args, clients):
        from utils.model_utils import load_model
        self.model, _ = load_model(args)
        super().__init__(args, clients)

        self.decay = getattr(args, 'decay', 0.9)
        self.alpha = getattr(args, 'alpha', 0.9)

        self.MAX_CONCURRENCY = 2
        self.client_queue = []
        self.staleness = {client.id: 0 for client in clients}
        self.cur_client = None
        self.wall_clock_time = 0
        self.round = 0

    def model2tensor(self):
        state_dict = self.model.state_dict()
        lora_params = {k: v for k, v in state_dict.items() if "lora_" in k}
        return lora_params

    def tensor2model(self, tensor_dict):
        current_state = self.model.state_dict()
        current_state.update(tensor_dict)
        self.model.load_state_dict(current_state)

    @abstractmethod
    def local_run(self):
        pass

    def run(self):
        raise NotImplementedError

    def sample(self):
        active_clients = [c for c in self.clients if c.status == Status.ACTIVE]
        if len(active_clients) >= self.MAX_CONCURRENCY:
            return []

        available_mem = self.get_available_cuda_memory()
        required_mem_per_client = 8.0  
        max_allowed_parallel = int(available_mem // required_mem_per_client)
        if max_allowed_parallel <= 0:
            return []  

        idle_clients = [c for c in self.clients if c.status == Status.IDLE]
        sample_num = min(
            self.MAX_CONCURRENCY - len(active_clients),
            len(idle_clients),
            max_allowed_parallel
        )
        if sample_num <= 0:
            return []

        sampled_clients = random.sample(idle_clients, sample_num)

        for client in sampled_clients:
            self.staleness[client.id] = 0

        return sampled_clients

    def downlink(self, clients):
        
        for client in clients:
            client.clone_model(self)

    # def client_update(self, clients):
    #     """客户端本地训练"""
    #     for client in clients:
    #         client.model.train()
    #         client.run()
    #         # 将完成训练的客户端加入队列，按完成时间排序
    #         heapq.heappush(self.client_queue, (self.wall_clock_time + client.training_time, client))
    #         client.status = Status.ACTIVE
    #
    #         del client.model  # 删除模型实例，释放权重占用
    #         client.model = None  # 置空，避免后续误引用
    #         torch.cuda.empty_cache()  # 清理 CUDA 缓存
    #         torch.cuda.synchronize()  # 等待缓存清理完成

    def uplink(self):
        if self.client_queue:
            completion_time, client_id = heapq.heappop(self.client_queue)
            self.cur_client = next(c for c in self.clients if c.id == client_id)
            self.wall_clock_time = completion_time
            return True
        return False

    def aggregate(self):
        import torch

        # debug summary
        server_lora = self.model2tensor() or {}
        client_lora = self.cur_client.model2tensor() or {}

        print(f"[AGG DEBUG] Server LoRA keys: {len(server_lora)}, Client LoRA keys: {len(client_lora)}")
        print("  Server sample keys:", list(server_lora.keys())[:6])
        print("  Client sample keys:", list(client_lora.keys())[:6])

        # Compute a quick check: if any common keys, compute average update norm
        common_keys = set(server_lora.keys()).intersection(set(client_lora.keys()))
        if len(common_keys) == 0:
            print("[AGG DEBUG] No common keys between server and client LoRA! aggregation likely no-op.")
        else:
            # compute average relative change norm on up to 5 keys
            samp = list(common_keys)[:5]
            norms = []
            for k in samp:
                s = server_lora[k]
                c = client_lora[k]
                try:
                    # ensure both tensors on same device & dtype
                    c = c.to(s.device).type_as(s)
                    d = (c - s).flatten()
                    norms.append(d.norm().item())
                except Exception as e:
                    print(f"[AGG DEBUG] norm calc failed for {k}: {e}")
            print(f"[AGG DEBUG] sample change norms for keys: {norms}")

        # 一次只对其中一个做

        def weight_decay(client_staleness):
            a = getattr(self.args, 'a', 1.0)
            b = getattr(self.args, 'b', 4.0)
            strategy = getattr(self.args, 'strategy', 'hinge')
            cur_round = self.round
            # ------- 轮流 decay -------
            if cur_round % 2 == 0:
                decay_target = "A"
                coeff = a
            else:
                decay_target = "B"
                coeff = b

            print(f"[Decay] Round {cur_round}, decaying {decay_target}, staleness={client_staleness}")

            # ----------- decay 计算 -----------
            s = max(1, client_staleness)  # 防止 0 影响

            if strategy == "hinge":
                decay = 1.0 / (1.0 + coeff * (s - 1))
            elif strategy == "poly":
                decay = 1.0 / ((s ** coeff) + 1)
            elif strategy == "exp":
                decay = math.exp(-coeff * s)
            else:  # constant
                decay = 1.0 / (1 + coeff)
            decay = float(max(min(decay, 1.0), 0.01))

            print(f"[Decay]  → decay factor = {decay:.5f}")

            return decay

        if self.cur_client is None:
            return

        client_staleness = self.staleness[self.cur_client.id]
        decay_factor = weight_decay(client_staleness)

        client_lora = self.cur_client.model2tensor() or {}
        server_lora = self.model2tensor() or {}

        server_keys = set(server_lora.keys())
        client_keys = set(client_lora.keys())
        missing_in_client = server_keys - client_keys
        extra_in_client = client_keys - server_keys
        if missing_in_client:
            print(
                f"[Aggregate] WARNING: {len(missing_in_client)} keys missing in client {self.cur_client.id} (showing up to 10):",
                list(missing_in_client)[:10])
        if extra_in_client:
            print(
                f"[Aggregate] NOTE: {len(extra_in_client)} extra keys present in client {self.cur_client.id} (showing up to 10):",
                list(extra_in_client)[:10])

        suffix_map = {}
        if missing_in_client:
            # build suffix->client key map (longest suffix first)
            client_key_list = list(client_keys)
            for sk in missing_in_client:
                found = None
                for ck in client_key_list:
                    # 如果 client key 的结尾与 server key 的结尾相同，则认为是对应项
                    if ck.endswith(sk) or sk.endswith(ck) or ck.split('.')[-3:] == sk.split('.')[-3:]:
                        found = ck
                        break
                if found:
                    suffix_map[sk] = found
            if suffix_map:
                print(f"[Aggregate] Suffix mapping applied for {len(suffix_map)} keys.")

        import torch
        aggregated_lora = {}
        # union of keys
        union_keys = server_keys.union(client_keys)
        for key in union_keys:
            if key in server_lora:
                server_val = server_lora[key]
            else:
                mapped_key = suffix_map.get(key, key)
                if mapped_key in client_lora:
                    server_val = torch.zeros_like(client_lora[mapped_key])
                else:
                    print(f"[Aggregate] Skipping key {key} (not in server nor mapped client).")
                    continue

            if key in client_lora:
                client_val = client_lora[key]
            else:
                mapped = suffix_map.get(key)
                if mapped and mapped in client_lora:
                    client_val = client_lora[mapped]
                else:
                    client_val = torch.zeros_like(server_val)

            try:
                client_val = client_val.to(server_val.device).type_as(server_val)
            except Exception:
                client_val = client_val.cpu().type_as(server_val.cpu())

            alpha_t = self.alpha * decay_factor
            aggregated_lora[key] = (1 - alpha_t) * server_val + alpha_t * client_val
        
        self.tensor2model(aggregated_lora)
        self.round += 1

    def update_status(self):
        if self.cur_client:
            self.cur_client.status = Status.IDLE

        for client in self.clients:
            if client.status == Status.ACTIVE:
                self.staleness[client.id] += 1

    def test_all(self):
        all_metrics = []
        for client in self.clients:
            try:
                original_state = {}
                if hasattr(client, 'model') and client.model is not None:
                    original_state = client.model2tensor()

                if hasattr(client, 'clone_model'):
                    client.clone_model(self)
                else:
                    client.model = copy.deepcopy(self.model)

                metrics = client.local_test(client.model)
                all_metrics.append(metrics)

                if original_state:
                    client.tensor2model(original_state)

            except Exception as e:
                print(f"Error testing client {client.id}: {e}")
                all_metrics.append({"eval_loss": 1.0, "perplexity": float("inf")})

        if not all_metrics:
            return {'loss': 1.0, 'perplexity': float("inf")}

        avg_loss = sum(m["eval_loss"] for m in all_metrics) / len(all_metrics)
        avg_perplexity = sum(m["perplexity"] for m in all_metrics) / len(all_metrics)
        return {'loss': avg_loss, 'perplexity': avg_perplexity}

    def get_available_cuda_memory(self):
        if not torch.cuda.is_available():
            return 0.0  

        device_index = 0  
        if hasattr(self.args, 'device'):
            device = self.args.device
            if isinstance(device, str):
                if device.startswith('cuda:'):
                    try:
                        device_index = int(device.split(':')[-1])
                    except ValueError:
                        device_index = 0  
                else:
                    return 0.0
            elif isinstance(device, int):
                device_index = device
            else:
                device_index = 0  

        max_valid_index = torch.cuda.device_count() - 1  
        if device_index < 0 or device_index > max_valid_index:
            device_index = 0  

        total_mem = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
        used_mem = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        available_mem = total_mem - used_mem
        return available_mem

    def client_update(self, clients):
        for client in clients:
            client.model.train()
            client.run()  

            completion_time = self.wall_clock_time + client.training_time
            heapq.heappush(self.client_queue, (completion_time, client.id))  
            client.status = Status.ACTIVE

            
            del client.model 
            client.model = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
