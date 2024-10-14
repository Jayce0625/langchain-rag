# 基于LangChain实现RAG检索应用

本repo的预训练大模型使用modelscope社区的Baichuan2-7B-Chat，结合langchain实现RAG检索增强。Linux服务器系统为Ubuntu 22.04，计算卡为NVIDIA A40-40g

## 依赖

1、构建并激活conda虚拟环境，使用Python3.10

```
conda create -n langchain python=3.10
conda activate langchain
```

2、确保CUDA已安装完毕（推荐CUDA12.1），若没有请执行

```
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
```

3、安装PyTorch-2.4.1（最新版本，本demo中并无bug）

```
# 使用清华源加速下载，实测从清华源下载的默认就是CUDA12.1的gpu版本pytorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

4、安装其余依赖，如langchain、modelscope、transformers等，向量数据库使用的是本地faiss-cpu向量数据库，用于存放我们文件中的内容

```
# 注意是modelscope[framework]，否则会丢失一些底层包
pip install langchain huggingface_hub pypdf modelscope[framework] transformers sentence_transformers faiss-cpu tiktoken accelerate bitsandbytes -i https://mirrors.aliyun.com/pypi/simple
```

## 用法

1、直接运行python脚本（无需第2步）

```
python langchain_rag.py
```

2、运行indexer.py，解析pdf生成向量库（仅生成并在本地存储faiss向量库）

```
python indexer.py
```
