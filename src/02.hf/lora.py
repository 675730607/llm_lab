import sys
import logging

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)s %(pathname)s:%(lineno)d  %(message)s",
)
logger = logging.getLogger(__name__)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from datasets import load_dataset
import transformers


param_model_path = "./asserts/foreign/models/Qwen3-0.6B-Base"
param_data_patth = "./asserts/foreign/datasets/BelleGroup/Belle_open_source_1K.json"


logger.debug("loading tokenizer")
tokenizer: transformers.models.Qwen2TokenizerFast = AutoTokenizer.from_pretrained(
    param_model_path,
    trust_remote_code=False,
)

special_token = dict(
    im_start=tokenizer("<|im-start|>").input_ids[0],
    im_end=tokenizer("<|im-end|>").input_ids[0],
    nl=tokenizer("\n").input_ids[0],
    IGNORE=tokenizer.pad_token_id,
    system=tokenizer("system").input_ids[0],
    user=tokenizer("user").input_ids[0],
    assistant=tokenizer("assistant").input_ids[0],
)

logger.debug("loading datasets")
ds = load_dataset(
    "json",
    data_files=param_data_patth,
    cache_dir="./.cache/huggingface/datasets",
)


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
                labels[i] = special_token["IGNORE"]
            i += 1

    attention_mask = labels.copy()
    for i in range(len(attention_mask)):
        if attention_mask[i] == special_token["IGNORE"]:
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
        max_length=1024,
    )


logger.debug("tokenizing datasets")
tokenized_ds = ds.map(
    tokenize_data,
    remove_columns=list(ds["train"].features),
    batched=False,
    load_from_cache_file=True,
    num_proc=10,
)
logger.debug("after tokenized train data feature %s", tokenized_ds)


logger.debug("loading model")
model = AutoModelForCausalLM.from_pretrained(
    param_model_path,
    trust_remote_code=False,
)


from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=None,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
logger.debug("changing to lora model")
model = get_peft_model(model, config)

training_args = TrainingArguments(
    "./output/models/lora",
    logging_steps=10,
    save_steps=100,
    num_train_epochs=1,
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
)

from torchdata.datapipes.iter import IterableWrapper

trainer = Trainer(
    args=training_args,
    model=model,
    processing_class=tokenizer,
    train_dataset=IterableWrapper(tokenized_ds["train"]),
    eval_dataset=None,
    data_collator=default_data_collator,
)


logger.info("begin training")
trainer.train()
logger.info("finish trainning")
trainer.save_model()
