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

# --- 1. 配置模型和分词器 ---

# 🔴 修改 1：使用 ModelScope 下载后的【本地模型路径】
model_name = "/models/Qwen2-7B-Instruct"

dataset_path = "./dataset/processed_data"
output_dir = "./qwen-7b-text2sql-adapter"

# --- 2. QLoRA 配置 ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# --- 3. LoRA 配置 ---
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

# --- 4. 加载模型和分词器 ---
print("正在加载模型和分词器...")

# 🔴 修改 2：本地 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    local_files_only=True
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 🔴 修改 3：本地 model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

# 🔴 修改 4：QLoRA 训练必须关闭 cache
model.config.use_cache = False

# 准备模型进行 k-bit 训练
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# --- 5. 加载数据集 ---
print("正在加载数据集...")
dataset = load_from_disk(dataset_path)

# --- 6. 配置训练参数 ---
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
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    evaluation_strategy="steps",
    eval_steps=50
)

# --- 7. 初始化并开始训练 ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

print("🚀 开始微调！")
trainer.train()

# --- 8. 保存最终的适配器 ---
print("✅ 微调完成，正在保存最终的适配器...")
trainer.save_model(output_dir)
