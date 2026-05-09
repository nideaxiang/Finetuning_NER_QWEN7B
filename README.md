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


#### 训练记录
- 训练记录1：绷不住了，一开始尝试用的0.5b的模型进行lora微调 在vGPU-48GB上居然能够内核爆掉很多次，然后调小了 Lora 秩和lora_alpha=16,一丁点儿微调的参数，
显存占用可以到 30G
<img width="198" height="89" alt="image" src="https://github.com/user-attachments/assets/62aa7884-a370-4ba6-a13a-edd8f7207b08" />

我发现可能是因为 多个 JSON 字符串直接拼接在一起 ， 导致原来的正则表达式在处理长文本时，匹配压力呈几何倍数增长  ，在验证时候特别慢
<img width="732" height="193" alt="image" src="https://github.com/user-attachments/assets/9e870185-a8e3-4925-93d3-b9ccdb09dd08" />

这样确实能训练完，也能出结果，但是效率极低，验证一次一个小时。
第500个step出的结果能够实现锁住格式，并且验证也正确
<img width="715" height="149" alt="image" src="https://github.com/user-attachments/assets/a9b39802-8bff-4aa3-9bf9-25cf4363fece" />

swan记录：https://swanlab.cn/@duan_daniel/autodl-tmp/runs/ma8c865glg9qe1yq4pund/chart
所以尝试改进——》

- 训练记录2：
    - 换训练器：将transformers的Trainer换为 Seq2SeqTrainingArguments
    - 开启生成模式： predict_with_generate=True
    - 每步评估结果直接传回 CPU ： eval_accumulation_steps=1
    - 优化一下metrics:我之前用的正则表达式的方法是“pattern = r'entity_text\s*:\s*"(.*?)",\s*entity_type\s*:\s*"(.*?)"'”
  ○ 换为： entity_re = re.compile(r'\"entity_text\":\s*\"(.*?)\",\s*\"entity_label\":\s*\"(.*?)\"')  
    - 小模型上的指标表现
 
  <img width="1568" height="444" alt="image" src="https://github.com/user-attachments/assets/dae6050e-8401-4fb8-818d-3673223a29a3" />

——》下面采用7b开始正式训练

- 训练记录3：
   - 模型：换用7b 
   -  量化：量化包 bitsandbytes 很难搞，和torch的版本老是不匹配,我就干脆直接使用7bAWQ的版本进行微调，这也是qlora，不然显卡吃不消
    - 结果又提示awq这个包很久没有维护了，要让用 optimum  这个包，结果安装了这个包也没有跑通，提示我要进行transformer的降级，一降级原来的很多包就失效了
——》直接用7B lora吧

训练记录4：
    - 多次下载报错：safetensors integrity check failed, the download may be incomplete, please try again.
    - 遂换之 hug的镜像网站不再使用 ModelScope 社区的模型
    - 图标记录在swanlab:https://swanlab.cn/@duan_daniel/autodl-tmp/runs/q0sv01cbirbcs52ql3wtg/chart
<img width="1942" height="945" alt="image" src="https://github.com/user-attachments/assets/23adb05c-305e-4b02-b876-26e64115d684" />

    
    - 验证集结果：<img width="1541" height="155" alt="image" src="https://github.com/user-attachments/assets/99d017f0-d5a5-4bdd-ac98-0e88b2f20ff1" />

    - 最终结果：TrainOutput(global_step=782, training_loss=0.17069725311168318, metrics={'train_runtime': 5739.7057, 'train_samples_per_second': 2.181, 'train_steps_per_second': 0.136, 'total_flos': 1.3559249039914906e+17, 'train_loss': 0.17069725311168318, 'epoch': 1.9987220447284346})
    -训练过程图表：https://swanlab.cn/@duan_daniel/autodl-tmp/runs/q0sv01cbirbcs52ql3wtg/chart
    - 推理能力验证：
  ○ 正样本
<img width="484" height="50" alt="image" src="https://github.com/user-attachments/assets/1d6b4ff8-86c5-4cb2-931b-6d517ad784ca" />

  ○ 负样本
<img width="327" height="75" alt="image" src="https://github.com/user-attachments/assets/2e407362-8a83-46ed-9bc6-5063188cf19e" />

● dev上的验证:
评估结果: {'eval_loss': 1.4311603307724, 'eval_model_preparation_time': 0.027, 'eval_方剂_f1': 0.5795053003117882, 'eval_中医诊断_f1': 0.6746987951343882, 'eval_中医证候_f1': 0.8384879724593192, 'eval_其他治疗_f1': 0.12844036696022218, 'eval_中医治疗_f1': 0.8457711442289547, 'eval_中药_f1': 0.49711723250582235, 'eval_西医治疗_f1': 0.75999999995136, 'eval_西医诊断_f1': 0.89811320749735, 'eval_临床表现_f1': 0.8641975308148826, 'eval_中医治则_f1': 0.6093749999566772, 'eval_f1': 0.6684931506403929, 'eval_precision': 0.5020576131687055, 'eval_recall': 0.9999999999999255, 'eval_runtime': 1229.6362, 'eval_samples_per_second': 0.534, 'eval_steps_per_second': 0.134}


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
├── train_7b.ipynb       🔥 训练 Notebook
____train_0.5b.ipynb        0.5b的版本
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
