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
model_name = "Qwen/Qwen2-7B-Instruct"  # 您也可以选择Qwen2，如 "Qwen/Qwen2-7B-Instruct"
dataset_path = "./processed_data"  # 上一步处理好的数据集路径
output_dir = "./qwen-7b-text2sql-adapter" # 微调后适配器权重保存目录

# --- 2. QLoRA 配置 ---
# 使用4-bit量化以节省显存
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
    # Qwen1.5 的目标模块，Qwen2 可能稍有不同，但通常是这些
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
)

# --- 4. 加载模型和分词器 ---
print("正在加载模型和分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# 解决Qwen的pad_token问题
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto", # 自动将模型分配到可用显卡
    trust_remote_code=True
)

# 准备模型进行k-bit训练
model = prepare_model_for_kbit_training(model)
# 应用LoRA配置
model = get_peft_model(model, peft_config)

# --- 5. 加载数据集 ---
print("正在加载数据集...")
dataset = load_from_disk(dataset_path)

# --- 6. 配置训练参数 ---
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,                     # 训练轮次
    per_device_train_batch_size=2,          # 每个GPU的批处理大小
    gradient_accumulation_steps=4,          # 梯度累积步数，有效批大小 = 2 * 4 = 8
    optim="paged_adamw_32bit",              # 使用分页优化器节省显存
    save_steps=50,                          # 每50步保存一次检查点
    logging_steps=10,                       # 每10步记录一次日志
    learning_rate=2e-4,                     # 学习率
    weight_decay=0.001,
    fp16=False,                             # 如果您的GPU支持，请使用bf16=True
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,                           # 如果设置为正数，则覆盖num_train_epochs
    warmup_ratio=0.03,                      # 预热比例
    group_by_length=True,                   # 按长度分组样本，提高训练效率
    lr_scheduler_type="cosine",             # 学习率调度器
    report_to="tensorboard",
    evaluation_strategy="steps",            # 每N步进行一次验证
    eval_steps=50                           # 每50步验证一次
)

# --- 7. 初始化并开始训练 ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=1024,                     # 最大序列长度
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,                          # 是否将多个短样本打包成一个长样本
)

print("🚀 开始微调！")
trainer.train()

# --- 8. 保存最终的适配器 ---
print("✅ 微调完成，正在保存最终的适配器...")
trainer.save_model(output_dir)