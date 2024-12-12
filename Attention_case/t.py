





from transformers import GPT2TokenizerFast, LlamaConfig, LlamaForCausalLM
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# 加载并更新tokenizer

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

# 保存tokenizer到本地目录
tokenizer.save_pretrained('./llama_tokenizer_directory')

tokenizer = GPT2TokenizerFast.from_pretrained('./llama_tokenizer_directory')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.save_pretrained('./llama_tokenizer_directory')

# 定义模型配置
config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=1024,
)

# 初始化模型并扩展词汇表
model = LlamaForCausalLM(config)
model.resize_token_embeddings(len(tokenizer))

# 加载数据集并进行预处理
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

def tokenize_function(examples):
    return tokenizer(examples['text'], return_special_tokens_mask=True, padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

import torch
# 确认输入数据的大小和形状
for batch in tokenized_datasets['train']:
    input_ids = torch.tensor(batch['input_ids'])
    print(input_ids.shape)
    break

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# 使用DataCollatorForLanguageModeling来动态填充输入序列
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
)

# 开始训练

import warnings
import traceback
def warn_with_traceback(m,c,f,l,file=None, line=None):
    log = f"{f}:{l}:{c.__name__}:{m}\n"
    log += "".join(traceback.format_stack())
    print(log)

warnings.showwarning = warn_with_traceback


trainer.train()


