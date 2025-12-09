import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def load_model(args):
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # auto / cuda:0 / cpu ...
        torch_dtype=torch.float16
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )

    return get_peft_model(model, lora_config), tokenizer

# def load_model(args):
#     model_name = args.model
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     # 补充：部分模型（如Qwen）需设置pad_token，避免训练警告（原有逻辑兼容）
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         device_map="auto",
#         torch_dtype=torch.float16
#     )
#
#     # -------------------------- 新增：FedSA-LoRA 动态配置 --------------------------
#     # 优先读取args参数（适配FedSA-LoRA），无则用原有默认值（兼容其他算法）
#     lora_rank = getattr(args, "lora_rank", 8)
#     lora_scaling = getattr(args, "lora_scaling", 32)
#     target_modules = ["q_proj", "v_proj"]  # 保持原有目标层，与论文一致
#
#     # rsLoRA特殊处理：缩放因子=scaling*rank（仅当变体为rslora时生效）
#     if getattr(args, "lora_variant", "lora") == "rslora":
#         lora_scaling *= lora_rank
#     # -----------------------------------------------------------------------------
#
#     lora_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         r=lora_rank,  # 动态适配（FedSA-LoRA从args读取，其他算法用默认8）
#         lora_alpha=lora_scaling,  # 动态适配（含rsLoRA处理）
#         lora_dropout=0.05,
#         target_modules=target_modules
#     )
#
#     return get_peft_model(model, lora_config), tokenizer

# def load_model(args):
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     import torch
#
#     # 关键参数：启用低显存模式
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_name_or_path,
#         torch_dtype=torch.float16,  # 半精度（显存占用减半）
#         low_cpu_mem_usage=True,  # 低 CPU 内存占用
#         device_map="auto",  # 自动设备映射（仅将必要层移到 GPU）
#         load_in_8bit=False,  # 可选：若支持 8bit 量化，设为 True（显存再减一半）
#         trust_remote_code=True,
#         # LoRA 专用：禁用梯度计算的层不加载到 GPU
#         offload_folder="./offload",  # 可选：将不常用层卸载到 CPU
#         offload_state_dict=True,
#     )
#
#     # 加载 tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
#     tokenizer.pad_token = tokenizer.eos_token  # 避免 pad token 错误
#
#     return model, tokenizer