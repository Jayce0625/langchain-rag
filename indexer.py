# 导入langchain相关包以实现RAG检索增强
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS

# 使用OCR解析pdf中图片里面的文字，并切成chunk片段，块大小为100，块重叠token数为10
pdf_loader_llm=PyPDFLoader('LLM.pdf', extract_images=True)   # 该文件讲述LLM大模型相关知识
chunks_llm=pdf_loader_llm.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10))  # 先load再split，chunk_size块大小，chunk_overlap分块时的重叠大小

pdf_loader_wjc=PyPDFLoader('wjc_jianli.pdf', extract_images=True)   # 该文件为我的个人简历综合资料
chunks_wjc=pdf_loader_wjc.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10))  # 先load再split，chunk_size块大小，chunk_overlap分块时的重叠大小

# 加载embedding模型，用于将chunk向量化
embeddings=ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_chinese-base')  # 因Linux服务器不好科学上网的原因，所以此处使用魔搭社区的开源词嵌入模型

# 构建并将chunk插入到faiss本地向量数据库（此处两个文件构建了两个数据库，其一是“LLM知识”的向量数据库，其二是“魏嘉辰个人简介”的向量数据库）
vector_db_llm=FAISS.from_documents(chunks_llm, embeddings)
vector_db_llm.save_local('LLM.faiss')

vector_db_wjc=FAISS.from_documents(chunks_wjc, embeddings)
vector_db_llm.save_local('WJC.faiss')

print('2 faiss saved!')
