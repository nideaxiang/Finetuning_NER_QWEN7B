# 🌿 中医药领域命名实体识别 (NER)

基于 **Qwen2.5-7B-Instruct** 模型进行微调的中医药领域命名实体识别任务 💊

## 📋 项目简介

本项目使用 **LoRA (Low-Rank Adaptation)** 技术对 Qwen2.5-7B-Instruct 大语言模型进行高效微调，实现中医药领域的精准命名实体识别。模型能够从文本中智能提取以下 **10 种**实体类型：

- 🥣 **方剂**：如 "麻黄汤"
- 🏥 **中医诊断**：如 "感冒"
- 📊 **中医证候**：如 "风热犯肺"
- 💉 **其他治疗**：如 "针灸"
- 🌡️ **中医治疗**：如 "清热解毒"
- 🌱 **中药**：如 "当归"、"柴胡"
- 💊 **西医治疗**：如 "抗生素治疗"
- 🩺 **西医诊断**：如 "肺炎"
- 😷 **临床表现**：如 "咳嗽"、"发热"
- 📝 **中医治则**：如 "疏风解表"


## 训练平台 Autodl :VGPU48 
八嘎雅鹿，中间训练不是很顺利，这个项目一共花了20元才跑完

## 🛠️ 技术栈

| 工具 | 说明 |
|------|------|
| 🔥 **PyTorch** | 深度学习框架 |
| 🧠 **Qwen2.5-7B-Instruct** | 基础大语言模型 |
| 🚀 **LoRA (PEFT)** | 参数高效微调方法 |
| 🤗 **Hugging Face Transformers** | 训练工具库 |
| 📈 **SwanLab** | 实验日志与可视化 |

## 🚀 快速开始

### 📦 环境要求

```bash
pip install torch transformers peft datasets pandas modelscope swanlab
```

### 📁 数据准备

数据集采用 **BIO 标注格式**，包含三个文件：
- `medical.train` - 训练集 (5259 条) 📚
- `medical.dev` - 验证集 (657 条) ✅  
- `medical.test` - 测试集 (658 条) 🧪

原始数据格式示例：
```
上 O
火 O
了 O
可 O
以 O
吃 O
点 O
柴 B-中药
胡 I-中药
```

### 🏃‍♂️ 训练

运行 `train_7.5b.ipynb` Notebook 进行模型微调：

1. **数据预处理** 🧹：将 BIO 格式转换为 JSONL 格式
2. **加载模型** 📥：加载 Qwen2.5-7B-Instruct 模型和分词器
3. **配置 LoRA** ⚙️：设置秩为 4，alpha 为 16
4. **开始训练** 🔥：使用 Hugging Face Trainer 进行训练

**训练参数**：
- ✨ 可训练参数：约 172 万 (仅 0.02% of 76 亿)
- 📏 最大序列长度：512
- 🎯 数据增强：加入 1000 条负面样本

### 🔮 推理

运行 `infer.ipynb` Notebook 进行推理：

```python
def predict(messages, model, tokenizer):
    model.eval()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response
```

**推理示例**：

**输入** 📥: 
```
文本:上火了可以吃点柴胡
```

**输出** 📤: 
```json
{"entity_text": "柴胡", "entity_label": "中药"}
```

### 🔗 模型合并

训练完成后，可以将 LoRA adapter 合并到原始模型中：

```python
model = PeftModel.from_pretrained(model, model_id=lora_adapter_path)
model = model.merge_and_unload()

output_path = "model/med_ner_qwen7b"
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
```

## 📂 项目结构

```
├── medical.train          📚 训练数据
├── medical.dev            ✅ 验证数据
├── medical.test           🧪 测试数据
├── ccf_train.jsonl        🚫 负面样本数据
├── train_7.5b.ipynb       🔥 训练 Notebook
├── infer.ipynb            🔮 推理 Notebook
├── train_raw.jsonl        📋 预处理后的训练数据
├── dev.jsonl              📋 预处理后的验证数据
├── test.jsonl             📋 预处理后的测试数据
├── output/                📦 训练输出目录
│   └── Qwen2.5-7b-NER-Fixed/
│       └── checkpoint-782/ 🏆 最佳检查点
└── model/
    └── med_ner_qwen7b/    🎯 合并后的完整模型
```

## 📊 指标计算

训练过程中使用自定义指标函数计算 **F1、精确率和召回率**，针对每个实体类型单独计算指标，确保全面评估模型性能 📈

## 📜 License

本项目仅供研究和学习使用 📚

## 🔗 参考框架

- 🧠 Qwen2.5: https://github.com/QwenLM/Qwen2.5
- 🚀 PEFT: https://github.com/huggingface/peft
- 🤗 Transformers: https://github.com/huggingface/transformers

---

⭐ 如果这个项目对你有帮助，请给个 Star 支持一下！
