---
title: "LLM Finetuning"
date: 2025-07-03
categories: [LLM]
tags: [LLM]
---
# 数据预处理
``` python
def generate_training_data(data_point):
    """
    将输入和输出文本转换为模型可读的 tokens。

    参数：
        data_point (dict): 包含 "instruction"、"input" 和 "output" 字段的字典

    返回：
        dict: 包含模型输入 IDs、标签和注意力掩码的字典
    """
    # 构建完整的输入prompt
    prompt = f"""\
[INST] <<SYS>>
You are a helpful assistant and good at writing Tang poem. 你是一個樂於助人的助手且擅長寫唐詩。
<</SYS>>

{data_point["instruction"]}
{data_point["input"]}
[/INST]"""

# 计算用户prompt的tokens数量
len_user_prompt_tokens = (
    len(
        tokenizer(
            prompt,
            truncation=True,
            max_length=CUTOFF_LEN+1,
            paddings="max_length",
        )["input_ids"]
    ) - 1
)

# 将完整的输入和输出转换为tokens
full_tokens = tokenizer(
    prompt + " " + data_point["output"] + "</s>",
    truncation=True,
    max_length=CUTOFF_LEN + 1,
    padding="max_length",
)["input_ids"][:-1] # 选取第一个到倒数第二个token的序列。作为模型输入，用来预测下一个token

return{
    "input_ids":full_tokens,
    "labels":[-100]*len_user_prompt_tokens+full_tokens[len_user_prompt_tokens:],
    "attention_mask":[1]*len(full_tokens),
}
```
**函数解释**：

- **目的**：将原始数据转换为模型可以处理的输入格式。
- **处理步骤**：
  - 构建提示词 `prompt`，包括系统信息和用户的指令。
  - 使用 `tokenizer` 将 `prompt` 转换为 token，并计算其长度。
  - 将完整的输入（提示词和输出）转换为 tokens。
  - 构建 `labels`，对于提示词部分的 tokens，使用 `-100`（在计算损失时会被忽略），对于输出部分的 tokens，保留实际的 token ID。

# 模型评估函数

``` python
def evaluate(instruction,generation_config,max_len,input_text="",verbose=True):
    """
    获取模型在给定输入下的生成结果。

    参数：
    - instruction: 描述任务的字符串。
    - generation_config: 模型生成配置。
    - max_len: 最大生成长度。
    - input_text: 输入文本，默认为空字符串。
    - verbose: 是否打印生成结果。

    返回：
    - output: 模型生成的文本。
    """
    # 构建完整的输入提示词
    prompt = f"""\
    [INST] <<SYS>>
    You are a helpful assistant and good at writing Tang poem. 你是一個樂於助人的助手且擅長寫唐詩。
    <</SYS>>

    {instruction}
    {input_text}
    [/INST]"""

    # 将提示词转换为所需的token格式
    inputs = tokenizer(prompt,return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    # 使用模型生成回复
    generation_output = model.generate(
        input_ids = input_ids,
        generation_config = genration.config,
        return_dict_in_generate = True,
        output_scores=True,
        max_new_tokens = max_len,
    )

    # 解码并打印生成的回复
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        output = output.split("[/INST]")[1].replace("</s>","").replace("<s>","").replace("Assistant","").strip()
        if verbose:
            print(output)
    
    return output
```

**函数解释**：

- **目的**：给定一个指令，使用模型生成对应的回复。
- **处理步骤**：
  - 构建提示词 `prompt`，包括系统信息和用户的指令。
  - 使用 `tokenizer` 将 `prompt` 转换为模型的输入格式。
  - 调用 `model.generate` 生成文本。
  - 对生成的序列进行解码，提取模型的输出部分。

# 加载模型
``` python
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# 从指定路径加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir, # 模型缓存目录
    quantization_config=nf4_config
    low_cpu_mem_usage=True, # 降低CPU内存占用，优化加载过程。
)

# 创建tokenizer并设置结束符号(eos_token)
logging.getLogger('transformers').setLevel(logging.ERROR) # 将transformers库的日志级别设置为ERROR，减少不必要的日志输出
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    add_eos_token=True,
    cache_dir=cache_dir,
    quantization=nf4_config
)
tokenizer.pad_token = tokenizer.eos_token # 将pad_token设置为eos_token，用于填充操作。

max_len = 128
generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.1,
    num_beams=1,
    top_p=0.3,
    no_repeat_ngram_size=3,
    pad_token_id=2,
)
```

**主要函数解释**：

- **BitsAndBytesConfig**：配置模型的量化设置，使用 4 位精度以节省显存。
- **AutoModelForCausalLM**：加载预训练的语言模型。
- **AutoTokenizer**：加载对应的分词器。
- **GenerationConfig**：设置文本生成时的参数，如温度、采样策略等。

# 设置用于微调的超参数
``` python
num_train_data = 1040  # 设置用于训练的数据量，最大值为5000。通常，训练数据越多越好，模型会见到更多样化的诗句，从而提高生成质量，但也会增加训练时间。
                      # 使用默认参数(1040)：微调大约需要25分钟，完整运行所有单元大约需要50分钟。
                      # 使用最大值(5000)：微调大约需要100分钟，完整运行所有单元大约需要120分钟。
        

output_dir = "./output"  # 设置作业结果输出目录。
ckpt_dir = "./exp1"  # 设置 model checkpoint 保存目录（如果想将 model checkpoints 保存到其他目录下，可以修改这里）。
num_epoch = 1  # 设置训练的总 Epoch 数（数值越高，训练时间越长，若使用免费版的 Colab 需要注意时间太长可能会断线，本地运行不需要担心）。
LEARNING_RATE = 3e-4  # 设置学习率

""" 建议不要更改此单元格中的代码 """

cache_dir = "./cache"  # 设置缓存目录路径
from_ckpt = False  # 是否从 checkpoint 加载模型权重，默认值为否
ckpt_name = None  # 加载特定 checkpoint 时使用的文件名，默认值为无
dataset_dir = "./GenAI-Hw5/Tang_training_data.json"  # 设置数据集目录或文件路径
logging_steps = 20  # 定义训练过程中每隔多少步骤输出一次日志
save_steps = 65  # 定义训练过程中每隔多少步骤保存一次模型
save_total_limit = 3  # 控制最多保留多少个模型 checkpoint
report_to = "none"  # 设置不上报实验指标，也可以设置为 "wandb"，此时需要获取对应的 API，见：https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/pull/5
MICRO_BATCH_SIZE = 4  # 定义微批次大小
BATCH_SIZE = 16  # 定义一个批次的大小
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE  # 计算每个微批次累积的梯度步骤
CUTOFF_LEN = 256  # 设置文本截断的最大长度
LORA_R = 8  # 设置 LORA（Layer-wise Random Attention）的 R 值
LORA_ALPHA = 16  # 设置 LORA 的 Alpha 值
LORA_DROPOUT = 0.05  # 设置 LORA 的 Dropout 率
VAL_SET_SIZE = 0  # 设置验证集的大小，默认值为无
TARGET_MODULES = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"]  # 设置目标模块，这些模块的权重将被保存为 checkpoint。
device_map = "auto"  # 设置设备映射，默认值为 "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))  # 获取环境变量 "WORLD_SIZE" 的值，若未设置则默认为 1
ddp = world_size != 1  # 根据 world_size 判断是否使用分布式数据处理(DDP)，若 world_size 为 1 则不使用 DDP
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
```

# 模型微调
## 模型加载与配置
``` python
if from_ckpt:
    model = PeftModel.from_pretrained(model,ckpt_name)
    model = prepare_model_for_kbit_training(model) # 调用prepare_model_for_kbit_training对量化模型进行预处理，使其适用于训练
```

``` python
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
```
- 配置LoRA：创建一个LoraConfig对象，设置LoRA的超参数：
  - r：控制LoRA的秩（rank），影响模型的适应能力。
  - lora_alpha：LoRA的alpha参数，控制LoRA层的缩放。
  - target_modules：指定要应用LoRA的模型模块。
  - lora_dropout：LoRA层的dropout率，用于正则化。
- 应用LoRA：使用get_peft_model将LoRA配置应用到模型。
## 分词器与数据处理

``` python
tokenizer.pad_token_id = 0

with open(dataset_dir,"r",encoding="utf-8") as f:
    data_json = json.load(f)

with open("tmp_dataset.json","w",encoding="utf-8") as f:
    json.dump(data_json[:num_train_data],f,indent=2,ensure_ascii=False)
# 加载到Hugging Face Dataset：使用load_dataset加载处理后的数据。
data = load_dataset("json",data_files="tmp_dataset.json",download_mode="force_redownload")
```

## 数据集划分与映射
``` python
if VAL_SET_SIZE > 0:
    train_val = data["train"].data_test_split(
        test_size=VAL_SET_SIZE,
        shuffle=True,
        seed=42,
    )
    train_data = train_val["train"].shuffle().map(generate_training_data)
    val_data = train_val["test"].shuffle().map(generate_training_data)
else:
    train_data = data['train'].shuffle().map(generate_training_data)
    val_data = None
```
## 模型训练
``` python
trainer = transformers.Trainer(
    model = model,
    train_dataset = train_data,
    val_dataset = val_data,
    args = transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=50,
        num_train_epochs=num_epoch,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        output_dir=ckpt_dir,
        save_total_limit=save_total_limit,
        ddp_find_unused_parameters=False if ddp else None,
        report_to=report_to,
    )
)
```

- **初始化Trainer**：使用`transformers.Trainer`配置和启动模型训练：
  - `model`：要训练的模型。
  - `train_dataset`和`eval_dataset`：训练集和验证集。
  - `TrainingArguments`：训练参数配置：
    - `per_device_train_batch_size`：每个设备上的训练批量大小。
    - `gradient_accumulation_steps`：梯度累积步数。
    - `warmup_steps`：预热步数。
    - `num_train_epochs`：训练轮数。
    - `learning_rate`：学习率。
    - `fp16`：是否使用混合精度训练。
    - 其他参数用于控制日志记录、保存策略等。
  - `data_collator`：用于数据批处理的收集器。

## 模型训练与保存
``` python
model.train()
model.save_pretrained(ckpt_dir)
```

## 模型输出解析



### 遍历测试数据
```python
for data in tqdm(test_data):
id = data['id']
print(f"Question {id}:\n{data['prompt']}")
```
- 提取当前样本的 `id`，通常用于标识问题。
- 打印问题编号和问题内容。

### 数据预处理
```python
inputs = tokenizer(data_formulate(data), return_tensors="pt").to('cuda')
```
- 使用 `tokenizer` 对数据进行预处理：
  - `data_formulate(data)`：对数据进行格式化处理，可能添加特殊标记（如 `[INST]`）。
  - `return_tensors="pt"`：返回 PyTorch 张量格式。
  - `.to('cuda')`：将数据移动到 GPU 上，加速计算。

### 设置生成配置
```python
generation_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=200,
    pad_token_id=tokenizer.pad_token_id
)
```
- 创建生成配置对象 `generation_config`：
  - `do_sample=False`：禁用采样，使用贪心解码（greedy decoding）。
  - `max_new_tokens=200`：限制生成文本的最大新 tokens 数为 200。
  - `pad_token_id=tokenizer.pad_token_id`：指定填充 token 的 ID。

### 生成文本
```python
output = model.generate(**inputs, generation_config=generation_config)
```
- 调用模型的 `generate` 方法生成文本：
  - `**inputs`：传入预处理后的输入数据。
  - `generation_config=generation_config`：传入生成配置。

### 解码和处理生成的文本
```python
output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0].split('[/INST] ')[1]

original_model_response.append(output_text)
```
- 使用 `tokenizer.batch_decode` 将生成的 tokens 解码为文本：
  - `skip_special_tokens=True`：忽略特殊标记（如 `[INST]`、`[/INST]` 等）。
  - `[0]`：获取第一个样本的生成文本。
  - `.split('[/INST] ')[1]`：提取 `[/INST]` 后的内容，即模型生成的回答部分。


# 测试
``` python
test_data_path = ""
output_path = os.path.join(output_dir,"results.txt")

# 配置模型的量化设置，使用 4 位精度
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# 从预训练模型加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=nf4_config
)

# 从预训练模型加载语言模型，使用量化配置并指定设备
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=nf4_config,
    device_map={'': 0},  # 指定使用的设备，GPU 0
    cache_dir=cache_dir
)

# 从 checkpoint 加载已保存的模型权重
model = PeftModel.from_pretrained(model, ckpt_name, device_map={'': 0})

```

``` python
results = []

# 设置生成配置，包括随机度、束搜索等参数
generation_config = GenerationConfig(
    do_sample=True,
    temperature=temperature,
    num_beams=1,
    top_p=top_p,
    # top_k=top_k,  # 如果需要使用 top-k，可以在此设置
    no_repeat_ngram_size=no_repeat_ngram_size,
    pad_token_id=2
)

with open(test_data_path,"r",encoding="utf-8") as f:
    test_datas = json.load(f)

with open(output_path,"w",encoding="utf-8") as f:
    for (i,test_data) in enumerate(test_datas):
        predict = evaluate(test_data["instruction"], generation_config, max_len, test_data["input"], verbose=False)
        f.write(f"{i+1}. " + test_data["input"] + predict + "\n")
        print(f"{i+1}. " + test_data["input"] + predict)
```