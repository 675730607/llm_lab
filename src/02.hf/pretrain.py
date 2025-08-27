import sys
import logging

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)s  %(pathname)s:%(lineno)d %(message)s",
)
logger = logging.getLogger(__name__)

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
    models,
)
import datasets
from datasets import load_dataset
from itertools import chain


MODEL_PATH = "./asserts/foreign/models/Qwen3-0.6B-Base"
PRETRAIN_DATA_PATHS = "./asserts/foreign/datasets/seq-monkey/mobvoi_seq_monkey_general_open_corpus_5K.jsonl"


tokenizer: models.Qwen2TokenizerFast = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)
logger.debug(f"created tokenizer. model_max_length %s", tokenizer)


ds: datasets.DatasetDict = load_dataset(
    "json",
    data_files=PRETRAIN_DATA_PATHS,
    cache_dir="./.cache/huggingface/datasets",
)
logger.debug(f"pretrain dataset keys {ds.keys()}")
logger.debug(f"pretrain dataset features {ds["train"].features}")
logger.debug(f"pretrain data exapmle {ds["train"][0]}")


def tokenize_pretain(datas):
    return tokenizer([text for text in datas["text"]])


tokenized_ds = ds.map(
    tokenize_pretain,
    batched=True,
    num_proc=10,
    batch_size=100,
    remove_columns=list(ds["train"].features),
    load_from_cache_file=True,
    desc="Running tokenize in dataset",
)
logger.debug(f"tokenized dataset keys {tokenized_ds.keys()}")
logger.debug(f"tokenized dataset features {tokenized_ds["train"].features}")
logger.debug(f"tokenized data exapmle {tokenized_ds["train"][0]}")


def group_input_ids(
    datas,
):
    block_size = 512

    # 将文本拼接起来
    concatenated_data = {k: list(chain(*datas[k])) for k in datas.keys()}
    # 计算拼接起来的整体长度
    total_length = len(concatenated_data[list(datas.keys())[0]])
    # 如果长度太长进行分块
    if total_length > block_size:
        total_length = (total_length // block_size) * block_size

    # 按 block_size 进行分块
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_data.items()
    }

    # CLM 任务，labels 和 input 是相同的
    result["labels"] = result["input_ids"].copy()
    logger.debug(
        f"group_input_ids total_length {total_length} type of labels %s type of type of labels[0][0] %s",
        type(result["labels"]),
        type(result["labels"][0][0]),
    )

    return result  # {k: torch.tensor(t) for k, t in result.items()}


# 文本拼接
lm_ds = tokenized_ds.map(
    group_input_ids,
    batched=True,
    batch_size=1000,
    num_proc=30,
    load_from_cache_file=True,
    desc="group tokenized dateset",
)
logger.debug(f"lm_ds keys {lm_ds.keys()}")
logger.debug(f"lm_ds features {lm_ds["train"].features}")
logger.debug(f"lm_ds exapmle {lm_ds["train"][0]}")


# 初始化模型
config: models.Qwen3Config = AutoConfig.from_pretrained(MODEL_PATH)
logger.debug(f"model config {config}")

model: models.qwen3.Qwen3ForCausalLM = AutoModelForCausalLM.from_config(
    config,
    trust_remote_code=True,
)
n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
logger.debug(f"created model size={n_params/2**20:.2f}M {model}")

# 模型训练参数
trainingArg = TrainingArguments(
    output_dir="./output/models/pretrain",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=100,
    num_train_epochs=1,
    learning_rate=1e-4,
)

from torchdata.datapipes.iter import IterableWrapper

# 训练器
trainer = Trainer(
    args=trainingArg,
    processing_class=tokenizer,
    model=model,
    train_dataset=IterableWrapper(lm_ds["train"]),
    eval_dataset=None,
    data_collator=default_data_collator,
)


logger.info("begin pretraining")
trainer.train()
logger.info("finish pretraining")
trainer.save_model()
