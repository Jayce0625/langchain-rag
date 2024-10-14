import os
import argparse
from operator import itemgetter

# 本地构建离线预训练模型相关包（transformers、torch等）
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # CausalLM即因果大模型，即施加了掩码自注意力的自回归生成式大模型
from transformers.generation.utils import GenerationConfig
from modelscope import snapshot_download, Model

# 检索增强相关包（langchain相关）
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS


# 命令行参数
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument('--faiss_db', type=str, default='LLM', help='FAISS database file name. In this demo, you can use \'LLM\' or \'WJC\'')
args = parser.parse_args()


# --------------------------------- 构建大模型，因科学上网原因，以Baichuan2-7B-Chat为例使用国产魔搭下载构建本地模型 ---------------------------------
model_dir = snapshot_download("baichuan-inc/Baichuan2-7B-Chat", revision='master')  # 下载预训练权重至本地（Linux中默认为~/.cache/modelscope）
model = Model.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)  # 从本地加载预训练权重，精度使用fp16
# messages = []
# messages.append({"role": "user", "content": "讲解一下“温故而知新”"})  # 构建prompt及角色
# response = model(messages)
# print(response)
# ----------------------------------------------------------------- 构建大模型 -----------------------------------------------------------------


# ------------------------------ 检索，使用OCR解析pdf中图片里面的文字，并切成chunk片段，块大小为100，块重叠token数为10 ------------------------------
pdf_loader=PyPDFLoader(f'{args.faiss_db}.pdf', extract_images=True)   # 该文件讲述LLM大模型相关知识
chunks=pdf_loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10))  # 先load再split，chunk_size块大小，chunk_overlap分块时的重叠大小

# 加载embedding模型，用于将chunk向量化
embeddings=ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_chinese-base')  # 因Linux服务器不好科学上网的原因，所以此处使用魔搭社区的开源词嵌入模型

# 构建并将chunk插入到faiss本地向量数据库（此处两个文件构建了两个数据库，其一是“LLM知识”的向量数据库，其二是“魏嘉辰个人简介”的向量数据库）
vector_db=FAISS.from_documents(chunks, embeddings)
vector_db.save_local(f'{args.faiss_db}.faiss')

print(f'{args.faiss_db}.faiss saved at {os.getcwd()}!')
# -------------------------------------------------------------------- 检索 --------------------------------------------------------------------


# ------------------------------------------------ 增强，用本地向量库增强大模型的领域知识与领域能力 ------------------------------------------------
# 加载由参数指定的faiss向量库，用于知识召回
vector_db=FAISS.load_local(f'{args.faiss_db}.faiss', embeddings, allow_dangerous_deserialization=True)

# 开始对话
while True:
    query = input('query:')

    result_simi = vector_db.similarity_search(query, k = 5)  # RAG的R过程，生成检索得到的TOP5相似度的chunk片
    source_knowledge= "\n".join([x.page_content for x in result_simi])

    # 使用RAG技术，用从向量数据库中检索得到的与query相似度TOP5的chunk片段增强prompt提示词
    augmented_prompt = f"""Using the contexts below, answer the query.

    contexts:
    {source_knowledge}

    query: {query}"""

    messages = []
    messages.append({"role": "user", "content": augmented_prompt})
    response = model(messages)
    print(response)
    print(response['response'])
# -------------------------------------------------------------------- 增强 --------------------------------------------------------------------