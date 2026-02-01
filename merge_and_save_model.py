import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 配置 ---
base_model_path = "Qwen/Qwen2-7B-Instruct"  
adapter_path = "./qwen-7b-text2sql-adapter/checkpoint-850" # 您训练好的LoRA适配器路径
# 新的、合并后模型的保存路径
merged_model_save_path = "./trained_model/qwen-7b-text2sql-merged-850"

# --- 加载基础模型和分词器 ---
print(f"正在从 '{base_model_path}' 加载基础模型...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu", # 先加载到CPU，避免占用过多显存
    trust_remote_code=True
)

# --- 加载并合并LoRA适配器 ---
print(f"正在从 '{adapter_path}' 加载LoRA适配器...")
model = PeftModel.from_pretrained(model, adapter_path)

print("正在合并LoRA权重...")
model = model.merge_and_unload()

# --- 保存完整的、合并后的模型和分词器 ---
print(f"正在将合并后的模型保存到 '{merged_model_save_path}'...")
model.save_pretrained(merged_model_save_path)
tokenizer.save_pretrained(merged_model_save_path)

print("✅ 模型合并并保存成功！")