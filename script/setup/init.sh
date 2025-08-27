#!/bin/bash
set -e

step=1
if [ $# -ge 1 ]; then
    step=$1
    echo "init from setp ${step}"
fi

# 初始化项目
# 1. 下载预训练数据 seq-monkey
# 2. 下载微调数据 BelleGroup
# 3. 下载模型 Qwen3-0.6B-Base
# 4. 安装依赖的 python 模块


# 1. 下载预训练数据 seq-monkey
if [ ${step} -le 1 ]; then
    out_dir='./asserts/foreign/datasets/seq-monkey'
    download_file="${out_dir}/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2"

    echo "downloading ${download_file}"
    mkdir -p "${out_dir}"
    curl -L --progress-bar -o "${download_file}" \
        'https://huggingface.co/datasets/dcdmm/seq-monkey-gen/resolve/main/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2?download=true'
    echo "untaring ${download_file}"
    pv "${download_file}" | tar -C "${out_dir}" -xjvf -
    rm "${download_file}"
fi


# 2. 下载微调数据 BelleGroup
if [ ${step} -le 2 ]; then
    out_dir='./asserts/foreign/datasets/BelleGroup'
    download_file="${out_dir}/Belle_open_source_0.5M.json"

    echo "downloading ${download_file}"
    mkdir -p ${out_dir}
    curl -L --progress-bar -o ${download_file} \
        'https://huggingface.co/datasets/BelleGroup/train_0.5M_CN/resolve/main/Belle_open_source_0.5M.json?download=true'
fi


# 3. 下载模型 Qwen3-0.6B-Base
if [ ${step} -le 3 ]; then
    out_dir='./asserts/foreign/models/Qwen3-0.6B-Base'
    download_files="config.json generation_config.json merges.txt model.safetensors tokenizer.json tokenizer_config.json vocab.json"

    mkdir -p ${out_dir}
    for file in ${download_files} ; do
        echo "downloading model file ${file}"
        curl -L --progress-bar -o ${out_dir}/${file} \
            "https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/main/${file}?download=true"
    done
fi


# 4. 安装依赖的 python 模块
if [ ${step} -le 4 ]; then
    echo "install depend python module"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirement.txt
fi