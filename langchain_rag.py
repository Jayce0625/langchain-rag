import os

# 本地构建离线预训练模型相关包（transformers、torch等）
import torch
from transformers import AutoModelForCausalLM, AytoTokenizer  # CausalLM即因果大模型，即施加了掩码自注意力的自回归生成式大模型
from transformers.generation.utils import GenerationConfig
from modelscope import snapshot_download, Model

# 检索增强相关包（langchain相关）
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------- 构建大模型，因科学上网原因，以Baichuan2-7B-Chat为例使用国产魔搭下载构建本地模型 ---------------------------------
model_dir = snapshot_download("baichuan-inc/Baichuan2-7B-Chat", revision='master')  # 下载预训练权重至本地（Linux中默认为~/.cache/modelscope）
model = Model.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)  # 从本地加载预训练权重，精度使用fp16
messages = []
messages.append({"role": "user", "content": "讲解一下“温故而知新”"})  # 构建prompt及角色
response = model(message)
print(response)
# ----------------------------------------------------------------- 构建大模型 -----------------------------------------------------------------

# # ------------------------------ 检索，使用OCR解析pdf中图片里面的文字，并切成chunk片段，块大小为100，块重叠token数为10 ------------------------------
# pdf_loader_llm=PyPDFLoader('LLM.pdf', extract_images=True)   # 该文件讲述LLM大模型相关知识
# chunks_llm=pdf_loader_llm.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10))  # 先load再split，chunk_size块大小，chunk_overlap分块时的重叠大小

# pdf_loader_wjc=PyPDFLoader('wjc_jianli.pdf', extract_images=True)   # 该文件为我的个人简历综合资料
# chunks_wjc=pdf_loader_wjc.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10))  # 先load再split，chunk_size块大小，chunk_overlap分块时的重叠大小

# # 加载embedding模型，用于将chunk向量化
# embeddings=ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_chinese-base')  # 因Linux服务器不好科学上网的原因，所以此处使用魔搭社区的开源词嵌入模型

# # 构建并将chunk插入到faiss本地向量数据库（此处两个文件构建了两个数据库，其一是“LLM知识”的向量数据库，其二是“魏嘉辰个人简介”的向量数据库）
# vector_db_llm=FAISS.from_documents(chunks_llm, embeddings)
# vector_db_llm.save_local('LLM.faiss')

# vector_db_wjc=FAISS.from_documents(chunks_wjc, embeddings)
# vector_db_llm.save_local('WJC.faiss')

# print('2 faiss saved!')
# # -------------------------------------------------------------------- 检索 --------------------------------------------------------------------