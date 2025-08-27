import sys
import logging

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)s  %(pathname)s:%(lineno)d  %(message)s ",
)
logger = logging.getLogger(__name__)

from transformers import (
    models,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from datasets import (
    DatasetDict,
    Dataset,
    load_dataset,
)
from torchdata.datapipes.iter import IterableWrapper

param_mode_path = "./asserts/foreign/models/Qwen3-0.6B-Base"
param_data_path = "./asserts/foreign/datasets/BelleGroup/Belle_open_source_1K.jsonl"


tokenizer: models.Qwen2TokenizerFast = AutoTokenizer.from_pretrained(
    param_mode_path,
    trust_remote_code=False,
)
logger.debug("created tokenizer %s", tokenizer)

# 有特殊含义的 token
special_token = dict(
    # BOS
    im_start=tokenizer("<|im_start|>").input_ids[0],
    # EOS
    im_end=tokenizer("<|im_end|>").input_ids[0],
    # PAD
    IGNORE_TOKEN_ID=tokenizer.pad_token_id,
    # 换行符
    nl_otken=tokenizer("\n").input_ids[0],
    # 角色标识符
    system=tokenizer("system").input_ids[0],
    user=tokenizer("user").input_ids[0],
    assistant=tokenizer("assistant").input_ids[0],
)
logger.debug("special_token %s", special_token)


ds: DatasetDict = load_dataset(
    "json",
    data_files=param_data_path,
    cache_dir="./.cache/huggingface/datasets",
)
logger.debug("loaded train data feature %s", ds)


def tokenize_sentences(
    sentences,
    max_length=1024,
):
    input_ids = tokenizer.apply_chat_template(
        sentences,
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )

    labels = input_ids.copy()
    needMask = True
    i = 0
    while i < len(labels) - 1:
        if labels[i] == special_token["im_start"]:
            # im_start|role|nl|...
            needMask = labels[i + 1] != special_token["assistant"]
            i += 3
        elif labels[i] == special_token["im_end"]:
            # im_end|nl|...
            i += 2
        else:
            if needMask:
                labels[i] = special_token["IGNORE_TOKEN_ID"]
            i += 1

    attention_mask = labels.copy()
    for i in range(len(attention_mask)):
        if attention_mask[i] == special_token["IGNORE_TOKEN_ID"]:
            attention_mask[i] = 0
        else:
            attention_mask[i] = 1

    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )


def tokenize_data(data):
    return tokenize_sentences(
        [
            dict(
                role="user",
                content=data["instruction"],
            ),
            dict(
                role="assistant",
                content=data["output"],
            ),
        ],
        max_length=512,
    )


tokenized_ds = ds.map(
    tokenize_data,
    remove_columns=list(ds["train"].features),
    batched=False,
    load_from_cache_file=True,
    num_proc=10,
)
logger.debug("after tokenized train data feature %s", tokenized_ds)


model = AutoModelForCausalLM.from_pretrained(
    param_mode_path,
    trust_remote_code=False,
)

training_args = TrainingArguments(
    output_dir="./output/models/finetune",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=100,
    num_train_epochs=1,
    learning_rate=1e-4,
)
trainer = Trainer(
    args=training_args,
    processing_class=tokenizer,
    model=model,
    train_dataset=tokenized_ds["train"],
    eval_dataset=None,
    data_collator=default_data_collator,
)

logger.info("begin training")
trainer.train()
logger.info("finish training")
trainer.save_model()
