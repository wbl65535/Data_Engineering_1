import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class VectorStore:
    def __init__(self, persist_directory="data/vector_db", embedding_model="paraphrase-multilingual-MiniLM-L12-v2"):
        """
        使用适合中文文本的多语言模型初始化向量存储
        """
        # 确保目录存在
        os.makedirs(persist_directory, exist_ok=True)
        
        # 初始化具有持久化功能的客户端
        self.client = chromadb.PersistentClient(
            path=persist_directory, 
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 设置本地模型缓存目录
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        print(f"模型缓存目录: {cache_dir}")
        
        # 配置 TRANSFORMERS_CACHE 环境变量
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['HF_HOME'] = cache_dir
        os.environ['HF_DATASETS_CACHE'] = cache_dir
        
        # 配置下载重试
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        # 设置超时时间较长
        timeout = 180  # 3分钟
        
        # 初始化适合中文文本的嵌入模型
        print(f"加载嵌入模型: {embedding_model}")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.model = SentenceTransformer(
                    embedding_model,
                    cache_folder=cache_dir
                )
                print("嵌入模型已加载")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    print(f"加载失败，{wait_time}秒后重试... ({e})")
                    time.sleep(wait_time)
                else:
                    print(f"无法加载模型，尝试使用离线模式...")
                    try:
                        # 尝试离线模式
                        self.model = SentenceTransformer(
                            embedding_model,
                            cache_folder=cache_dir,
                            use_auth_token=False
                        )
                        print("使用离线模式加载模型成功")
                    except Exception as offline_error:
                        print(f"离线模式也失败: {offline_error}")
                        raise
        
        # 集合名称
        self.collection_name = "intelligent_data_engineering"
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
        
        print(f"向量存储已初始化，路径: {persist_directory}")
    
    def generate_embeddings(self, texts):
        """为文本列表生成嵌入向量"""
        return self.model.encode(texts).tolist()
    
    def reset_collection(self):
        """重置向量存储，删除所有现有文档"""
        if self.collection.count() > 0:
            print(f"正在重置向量存储，删除 {self.collection.count()} 个文档...")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )
            print("向量存储已重置")
        else:
            print("向量存储为空，无需重置")
    
    def add_documents(self, chunks):
        """
        将文档块添加到向量存储
        chunks: 包含'text'和'metadata'键的字典列表
        """
        # 检查集合中是否已有文档
        if self.collection.count() > 0:
            print(f"集合中已包含 {self.collection.count()} 个文档")
            return
        
        print(f"正在向向量存储添加 {len(chunks)} 个块...")
        
        ids = []
        texts = []
        metadatas = []
        
        for i, chunk in enumerate(tqdm(chunks)):
            # 生成唯一ID
            chunk_id = f"chunk_{i}"
            
            ids.append(chunk_id)
            texts.append(chunk["text"])
            
            # 将元数据转换为字符串以兼容ChromaDB，但保持原始值
            metadata = {}
            for k, v in chunk["metadata"].items():
                # 确保数值类型的元数据被正确转换为字符串
                if isinstance(v, (int, float)):
                    metadata[k] = str(v)
                else:
                    metadata[k] = str(v)
            metadatas.append(metadata)
        
        # 生成嵌入向量并添加到集合
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"已添加 {len(chunks)} 个块到向量存储")
        print(f"向量存储现在包含 {self.collection.count()} 个文档")
    
    def similarity_search(self, query, top_k=5):
        """
        使用查询进行相似度搜索
        返回最相似的文档及其元数据
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        # 格式化结果
        formatted_results = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            formatted_results.append({
                "text": doc,
                "metadata": meta,
                "similarity": 1 - dist  # 将距离转换为相似度分数(0-1)
            })
        
        return formatted_results
    
    def get_stats(self):
        """返回向量存储的统计信息"""
        return {
            "document_count": self.collection.count(),
            "collection_name": self.collection_name
        }
    
    def save_content_for_inspection(self, output_path="data/vector_content.json"):
        """将向量存储内容保存为JSON文件以便检查"""
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 获取所有文档
        results = self.collection.get()
        
        documents = []
        for i in range(len(results['ids'])):
            documents.append({
                "id": results['ids'][i],
                "text": results['documents'][i],
                "metadata": results['metadatas'][i]
            })
        
        # 保存为JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        print(f"已将向量存储内容保存到 {output_path} 以便检查")
        return documents