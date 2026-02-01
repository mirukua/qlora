import json
import csv
from tqdm import tqdm
from vllm import LLM, SamplingParams

# --- 1. 配置 ---
# 关键改动：模型路径现在指向我们上一步保存的、已合并的文件夹
merged_model_path = "./trained_model/qwen-7b-text2sql-merged-850"
test_data_path = "./dataset/test.json"
output_csv_path = "submission_vllm.csv"
# 根据您的GPU显存大小设置，如果显存大，可以设为1.0
gpu_memory_utilization = 0.90 

# --- 2. 加载测试数据并准备所有Prompts ---
print(f"正在从 '{test_data_path}' 加载测试数据...")
with open(test_data_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

prompt_template = """<s>[INST] 你是一个SQL专家。请根据以下问题，生成相应的SQLite查询语句。

### 问题:
{question} [/INST]

### SQL:
"""

# vLLM的核心优势：一次性准备好所有prompts
prompts = [prompt_template.format(question=item['NL']) for item in test_data]
ids = [item['id'] for item in test_data]

# --- 3. 初始化vLLM模型和采样参数 ---
print(f"正在从 '{merged_model_path}' 初始化vLLM引擎...")
# tensor_parallel_size可以根据您的GPU数量设置，单卡设为1
llm = LLM(model=merged_model_path, 
          trust_remote_code=True, 
          gpu_memory_utilization=gpu_memory_utilization,
          tensor_parallel_size=1)

# 配置采样参数，对应您原来 model.generate() 中的设置
# do_sample=False 对应 temperature=0
sampling_params = SamplingParams(
    n=1,
    temperature=0,
    max_tokens=256, # 等同于 max_new_tokens
    # stop_token_ids=[tokenizer.eos_token_id], # 可以指定停止token，vLLM会自动处理
)

# --- 4. 执行批量推理 ---
print(f"开始对 {len(prompts)} 条数据进行高速批量推理...")
# 这是vLLM最核心的一步，一次调用，处理所有prompts！
outputs = llm.generate(prompts, sampling_params)

# --- 5. 处理并保存结果 ---
results = []
full_results = []
print("正在处理生成结果...")
for i, output in enumerate(tqdm(outputs)):
    item_id = ids[i]
    generated_text = output.outputs[0].text
    full_results.append(generated_text)
    
    # 提取SQL部分（这里的逻辑和您原来的一样）
    try:
        sql_part = generated_text.strip()
        if sql_part.endswith("</s>"):
             sql_part = sql_part[:-len("</s>")].strip()
        pred_sql = sql_part
    except IndexError:
        pred_sql = ""
        
    results.append({"id": item_id, "pred_sql": pred_sql})

# --- 保存为CSV ---
print(f"推理完成，正在保存到 {output_csv_path}...")
with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['id', 'pred_sql']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

full_results = '\n'.join(full_results)
with open("./result.txt", 'w', encoding='utf-8') as f:
    f.write(full_results)

print(f"✅ 所有任务完成！结果已保存至 {output_csv_path}")