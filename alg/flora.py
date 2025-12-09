# flora.py —— FLoRA 极简版，完全模仿 fedit 风格

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record


# =============================
#   FLoRA 客户端
# =============================
class Client(FTBaseClient):
    """
    继承 FTBaseClient，只需要覆盖 LoRA 秩即可。
    所有训练流程依旧调用父类 run()，
    父类内部会：
        - 加载模型
        - 冻结 backbone
        - 只训练 LoRA
        - 记录时间
        - 取出 LoRA delta
    """

    def __init__(self, id, args):
        # 调用父类初始化（含 tokenizer, data, training_args）
        super().__init__(id, args)

        # ====== FLoRA 的关键差异：每个 client 用自己的 rank ======
        # config.yaml 中设置 local_rank 或 client-specific rank
        if hasattr(args, "local_rank"):
            self.lora_config.r = args.local_rank   # 简单 FLoRA（同质化）
        elif hasattr(args, f"client{id}_rank"):
            self.lora_config.r = getattr(args, f"client{id}_rank")  # 真正异构 FLoRA

    @time_record
    def run(self, model):
        """
        直接复用 FTBaseClient.run(model)
        FLoRA 所需逻辑都已由 self.lora_config 覆盖
        """
        super().run(model)



# =============================
#     FLoRA 服务器
# =============================
class Server(FTBaseServer):
    """
    完整复用 FTBaseServer:
        - sample()
        - local_run()
        - aggregate() (FedAvg for LoRA)
    FLoRA 不需要修改此部分
    """
    pass
