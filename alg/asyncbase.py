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
        self.training_time = random.uniform(1.0, 5.0)  # 模拟训练时间差异
        self.model = None  # 确保model属性存在

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
        # 先初始化模型，再调用父类
        from utils.model_utils import load_model
        self.model, _ = load_model(args)
        super().__init__(args, clients)

        self.decay = getattr(args, 'decay', 0.9)

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
        required_mem_per_client = 8.0  # 单客户端模型+训练的预估显存（GiB）
        max_allowed_parallel = int(available_mem // required_mem_per_client)
        if max_allowed_parallel <= 0:
            return []  # 显存不足，不采样新客户端

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
        """下发模型到客户端"""
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
        """从队列中获取最早完成的客户端"""
        if self.client_queue:
            completion_time, client_id = heapq.heappop(self.client_queue)
            self.cur_client = next(c for c in self.clients if c.id == client_id)
            self.wall_clock_time = completion_time
            return True
        return False

    def aggregate(self):
        """聚合客户端更新，考虑陈旧度。已做健壮处理：容错缺失 key 与后缀匹配。"""
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

        # 获取客户端和服务器模型的LoRA参数（字典形式）
        client_lora = self.cur_client.model2tensor() or {}
        server_lora = self.model2tensor() or {}

        # 一些简单的检查输出，便于调试
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

        # 尝试建立后缀映射：当 key 不在 client 中时，用其后缀在 client_keys 中模糊匹配
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

        # 聚合：对 union keys 遍历，缺失用零张量填充，或用后缀映射
        import torch
        aggregated_lora = {}
        # union of keys
        union_keys = server_keys.union(client_keys)
        for key in union_keys:
            # 得到 server tensor（若没有，从 client 推断形状并创建 zeros）
            if key in server_lora:
                server_val = server_lora[key]
            else:
                # server 缺失该 key，则尝试用 client 推断 shape
                mapped_key = suffix_map.get(key, key)
                if mapped_key in client_lora:
                    server_val = torch.zeros_like(client_lora[mapped_key])
                else:
                    # 作为兜底，跳过不合理键
                    print(f"[Aggregate] Skipping key {key} (not in server nor mapped client).")
                    continue

            # 得到 client tensor（若没有，尝试后缀映射或用 zeros）
            if key in client_lora:
                client_val = client_lora[key]
            else:
                mapped = suffix_map.get(key)
                if mapped and mapped in client_lora:
                    client_val = client_lora[mapped]
                else:
                    # client 没有该 key，用 zeros（与 server_val 同 shape/device/dtype）
                    client_val = torch.zeros_like(server_val)

            # 确保两者类型/设备匹配
            try:
                client_val = client_val.to(server_val.device).type_as(server_val)
            except Exception:
                # 容错：如果转换失败，尝试把 server_val 移到 cpu 并用 cpu 0 张量
                client_val = client_val.cpu().type_as(server_val.cpu())

            # 加权融合（注意 server_val 和 client_val 都是 tensor）
            aggregated_lora[key] = (
                        self.decay * decay_factor * client_val + (1 - self.decay * decay_factor) * server_val)

        # 更新服务器模型
        self.tensor2model(aggregated_lora)
        self.round += 1

    def update_status(self):
        """更新客户端状态和陈旧度"""
        if self.cur_client:
            self.cur_client.status = Status.IDLE

        # 更新所有活跃客户端的陈旧度
        for client in self.clients:
            if client.status == Status.ACTIVE:
                self.staleness[client.id] += 1

    def test_all(self):
        """测试所有客户端"""
        all_metrics = []
        for client in self.clients:
            try:
                # 保存客户端当前状态（如果模型存在）
                original_state = {}
                if hasattr(client, 'model') and client.model is not None:
                    original_state = client.model2tensor()

                # 使用服务器模型测试
                if hasattr(client, 'clone_model'):
                    client.clone_model(self)
                else:
                    # 如果客户端没有 clone_model 方法，直接设置模型
                    client.model = copy.deepcopy(self.model)

                metrics = client.local_test(client.model)
                all_metrics.append(metrics)

                # 恢复客户端状态
                if original_state:
                    client.tensor2model(original_state)

            except Exception as e:
                print(f"Error testing client {client.id}: {e}")
                # 添加默认指标以避免崩溃
                all_metrics.append({"eval_loss": 1.0, "perplexity": float("inf")})

        if not all_metrics:
            return {'loss': 1.0, 'perplexity': float("inf")}

        avg_loss = sum(m["eval_loss"] for m in all_metrics) / len(all_metrics)
        avg_perplexity = sum(m["perplexity"] for m in all_metrics) / len(all_metrics)
        return {'loss': avg_loss, 'perplexity': avg_perplexity}

    def get_available_cuda_memory(self):
        """获取 GPU 剩余显存（GiB），添加设备有效性校验"""
        if not torch.cuda.is_available():
            return 0.0  # 无 GPU 时返回 0

        # 步骤1：解析并校验 device_index
        device_index = 0  # 默认使用第 0 张 GPU
        if hasattr(self.args, 'device'):
            # 处理 args.device 的多种可能格式（如 'cuda:0'、0、'cpu'）
            device = self.args.device
            if isinstance(device, str):
                # 若为字符串（如 'cuda:0'、'cpu'），提取 GPU 索引
                if device.startswith('cuda:'):
                    try:
                        device_index = int(device.split(':')[-1])
                    except ValueError:
                        device_index = 0  # 解析失败默认用 0
                else:
                    # 若为 'cpu' 或其他字符串，直接返回 0（不使用 GPU）
                    return 0.0
            elif isinstance(device, int):
                # 若为整数，直接作为索引
                device_index = device
            else:
                device_index = 0  # 其他类型默认用 0

        # 步骤2：校验设备索引是否有效（不超过可用 GPU 数量）
        max_valid_index = torch.cuda.device_count() - 1  # 最大有效索引（从 0 开始）
        if device_index < 0 or device_index > max_valid_index:
            device_index = 0  # 无效索引默认用 0

        # 步骤3：计算剩余显存
        total_mem = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
        used_mem = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        available_mem = total_mem - used_mem
        return available_mem

    def client_update(self, clients):
        """客户端本地训练（异步入队+立即释放）"""
        for client in clients:
            client.model.train()
            client.run()  # 执行训练（训练后已 del model）

            # ！！！关键修改：仅入队「客户端 ID+完成时间」，不持有客户端实例！！！
            completion_time = self.wall_clock_time + client.training_time
            heapq.heappush(self.client_queue, (completion_time, client.id))  # 只存 ID，不存 client
            client.status = Status.ACTIVE

            # 立即释放客户端的所有临时资源
            del client.model  # 双重保险（客户端 run 中已 del，但这里再确认）
            client.model = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()