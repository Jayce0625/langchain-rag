import os
import sys
import argparse
import warnings
import logging

# 忽略所有警告及禁用所有日志信息
warnings.filterwarnings('ignore')
logging.disable(sys.maxsize)

# 本地构建离线预训练模型相关包（transformers、torch等）
import torch
from transformers import TextStreamer  # 流式输出
from transformers.generation.utils import GenerationConfig
from modelscope import snapshot_download, Model, AutoTokenizer, AutoModelForCausalLM

# 检索增强相关包（langchain相关）
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS


# 命令行参数
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument('--faiss_db', type=str, default='LLM', help='FAISS database file name. In this demo, you can use \'LLM\' or \'WJC\', \
                                                                 also your own pdf, just make sure it\'s in the root directory of this demo!')  # 
parser.add_argument('--benchmark', action='store_true', help='Comparing the output of LLM w/ and w/o RAG.')
args = parser.parse_args()


def stream_generate(model, messages, tokenizer):
    """
    将输入的prompt使用模型的tokenizer进行格式化, 并使用流式推理方式来逐token输出。

    args:
        model: 执行推理的LLM。
        messages: 需要被格式化及传入LLM执行推理的输入。
        tokenizer: 分词器。
        device: GPU索引。
    Return:
        新生成token的ID列表。
    """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # 使用分词器的apply_chat_template方法来格式化消息
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)  # 将格式化后的文本转换为模型输入，并转换为PyTorch张量，然后移动到指定的设备

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)  # 启动流式输出
    generated_ids = model.generate(**model_inputs, max_new_tokens=512, streamer=streamer)  # 前向推理，流式输出
    # 从生成的ID中提取新生成的ID部分
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    print("LLM response: ", end="")
    
    return generated_ids


# --------------------------------- 构建大模型，因科学上网原因，以Baichuan2-7B-Chat为例使用国产魔搭下载构建本地模型 ---------------------------------
model_dir = snapshot_download("baichuan-inc/Baichuan2-7B-Chat", revision='master')  # 下载预训练权重至本地（Linux中默认为~/.cache/modelscope）
# model = Model.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16).to(device)  # 从本地加载预训练权重，精度使用fp16
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)  # 加载分词器
# 设置聊天模板
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %} \
    {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
# ----------------------------------------------------------------- 构建大模型 -----------------------------------------------------------------

# -------------------------------------------------------------- 无 RAG 增强对话 --------------------------------------------------------------
if args.benchmark:
    print("\n================== \033[91mLLM without RAG!!!\033[0m ==================")  # 红色字体醒目提示
    query = input('query: ')

    # 格式化输入query并启用流式输出
    messages = [
        {"role": "system", "content": "You are Baichuan. You are a helpful assistant."},
        {"role": "user", "content": query},
    ]  # 构建prompt和角色
    
    for name, param in model.named_parameters():
        print(name, param.device)

    generated_ids = stream_generate(model, messages, tokenizer)  # 对输入进行格式化，执行流式推理
    print("LLM response: ", end="")
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]  # 使用分词器的batch_decode方法将生成的ID解码回文本，并跳过特殊token
    print(f"LLM response: \033[92m{response}\033[0m\n")

    # 一次性推理输出
    # response = model(messages)  # 前向推理
    # llm_response = response['response']  # 从response字典中提取出大模型的回复
    # print(f"LLM response: \033[92m{llm_response}\033[0m\n")  # 大模型输出绿色高亮
# --------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------- RAG检索，使用OCR解析pdf中图片里面的文字，并切成chunk片段，块大小为100，块重叠token数为10 -----------------------------
pdf_loader=PyPDFLoader(f'{args.faiss_db}.pdf', extract_images=True)   # 该文件讲述LLM大模型相关知识，仅定义加载器，此时还未真正加载pdf
# 下面才真正加载pdf，同时切分成段
chunks=pdf_loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10))  # 先load再split，chunk_size块大小，chunk_overlap分块时的重叠大小

# 加载中文词嵌入embedding模型，用于将chunk向量化
embeddings=ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_chinese-base')  # 因Linux服务器不好科学上网的原因，所以此处使用魔搭社区的开源词嵌入模型

# 构建并将chunk插入到faiss本地向量数据库
vector_db=FAISS.from_documents(chunks, embeddings)
vector_db.save_local(f'{args.faiss_db}.faiss')

print(f'\nThe faiss database {args.faiss_db}.faiss is saved in {os.getcwd()}/{args.faiss_db}.faiss !')
# ----------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------- RAG增强，用本地向量库增强大模型的领域知识与领域能力 -----------------------------------------------
# 加载由参数指定的faiss向量库，用于知识召回
vector_db=FAISS.load_local(f'{args.faiss_db}.faiss', embeddings, allow_dangerous_deserialization=True)

# 开始循环对话
print("\n================== \033[91mLLM with RAG!!!\033[0m ==================")
while True:
    query = input('query_with_rag: ')

    result_simi = vector_db.similarity_search(query, k = 5)  # RAG的R过程，生成检索得到的TOP5相似度的chunk片
    source_knowledge= "\n".join([x.page_content for x in result_simi])  # 将相似度高的片段进行拼接

    # 使用RAG技术，用从向量数据库中检索得到的与query相似度TOP5的chunk片段增强prompt提示词
    augmented_prompt = f"""Using the contexts below, answer the query.

    contexts:
    {source_knowledge}

    query: {query}"""

    messages = [
        {"role": "system", "content": "You are Baichuan. You are a helpful assistant."},
        {"role": "user", "content": augmented_prompt},
    ]  # 根据RAG增强得到的prompt构建用户输入

    generated_ids = stream_generate(model, messages, tokenizer)  # 对输入进行格式化，执行流式推理
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]  # 使用分词器的batch_decode方法将生成的ID解码回文本，并跳过特殊token

    # 一次性推理输出
    # response = model(messages)  # 执行推理
    # llm_response = response['response']  # 从response字典中提取出大模型的回复
    # print(response)  # 输出response，其是一个字典，包括response：模型回复; history：历史对话信息，history又包括每一轮对话相似度提取召回的content以及该轮的query
    # print(f"LLM_with_rag response: \033[92m{llm_response}\033[0m\n")  # 直接输出大模型的回复
    
    # 若给定命令行参数benchmark则只对话一次用于对比RAG效果，或者是在对话中输入bye或再见则中断对话
    if args.benchmark or ("bye" in query.lower() or "再见" in query.lower()):
        break
# ------------------------------------------------------------------- RAG增强 -------------------------------------------------------------------