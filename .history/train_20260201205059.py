import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ==============================
# 0️⃣ 强制 HuggingFace 离线模式（推荐）
# ==============================
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ==============================
# 1️⃣ 本地模型 & 数据路径
# ==============================
model_path = r"C:\Users\DELL\.cache\modelscope\hub\models\Qwen\Qwen2.5-7B-Instruct"

dataset_path = "./processed_data"
output_dir = "./qwen-7b-text2sql-adapter"

# ==============================
# 2️⃣ QLoRA 4-bit 配置
# ==============================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# ==============================
# 3️⃣ LoRA 配置
# ==============================
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

# ==============================
# 4️⃣ 加载 tokenizer（完全本地）
# ==============================
print("📦 正在从本地加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True      # ✅ 关键
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ==============================
# 5️⃣ 加载模型（完全本地）
# ==============================
print("📦 正在从本地加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True      # ✅ 关键
)

# QLoRA 训练准备
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# ==============================
# 6️⃣ 加载数据集（本地）
# ==============================
print("📚 正在加载数据集...")
dataset = load_from_disk(dataset_path)

# ==============================
# 7️⃣ 训练参数
# ==============================
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    bf16=True,
    fp16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    evaluation_strategy="steps",
    eval_steps=50,
)

# ==============================
# 8️⃣ Trainer
# ==============================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=1024,
    args=training_arguments,
    packing=False,
)

print("🚀 开始微调（完全离线）...")
trainer.train()

print("✅ 保存 LoRA 适配器...")
trainer.save_model(output_dir)
