import os
import json
from dotenv import load_dotenv
import httpx
from typing import List, Dict, Any

class QASystem:
    def __init__(self, api_key=None):
        """
        使用SiliconFlow API集成初始化问答系统
        """
        # 如果未提供API密钥，则从环境变量加载
        load_dotenv()
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = "https://api.siliconflow.cn/v1"
        
        if not self.api_key:
            print("警告: 未找到API密钥。请设置API_KEY环境变量。")
        else:
            print("API配置已初始化")
    
    def format_context(self, retrieved_documents: List[Dict[Any, Any]]) -> str:
        """
        将检索到的文档格式化为模型的上下文字符串
        """
        context_parts = []
        
        for i, doc in enumerate(retrieved_documents):
            source = doc["metadata"].get("source", "Unknown")
            page = doc["metadata"].get("page_number", "Unknown")
            paragraph = doc["metadata"].get("paragraph_number", "Unknown")
            
            # 格式化引用
            ref = f"[来源{i+1}: 文档《{source}》第{page}页第{paragraph}段]"
            
            # 添加格式化的文档
            context_parts.append(f"{ref}\n{doc['text']}\n")
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, retrieved_documents: List[Dict[Any, Any]]) -> str:
        """
        根据查询和检索到的文档生成答案
        """
        if not self.api_key:
            return "错误: API密钥未设置。请设置API_KEY环境变量。"
        
        # 格式化上下文
        context = self.format_context(retrieved_documents)
        
        # 格式化完整提示
        system_prompt = """你是一个智能数据工程课程的助手。根据提供的参考文档回答用户问题。
        - 只使用提供的参考文档中的信息回答问题，不要使用其他知识。
        - 即使文档中没有直接明确标注主题的段落，也要从内容中提取和分析相关信息。
        - 当信息分散在多个文档中时，请综合分析并给出完整回答。
        - 如果问题涉及多方面内容，请尽量全面地从文档中找出相关信息进行回答。
        - 如果参考文档中完全没有相关信息，直接说明无法回答该问题，同时不需给出信息来源。
        - 回答要全面、准确，并始终在回答末尾注明信息来源（包括文档名、页码和段落）。
        - 如果使用了多个参考来源，请分别标明各个来源。"""
        
        user_prompt = f"""参考文档：
        {context}
        
        用户问题：{query}
        
        请根据以上参考文档回答问题，并在回答末尾注明信息来源："""
        
        # print("正在发送请求到API...")
        
        try:
            # 直接使用httpx而不是OpenAI客户端
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "Qwen/Qwen2.5-72B-Instruct",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 2048,
                "stream": False  # 使用非流式以简化处理
            }
            
            # 发送请求
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                # 检查响应状态
                response.raise_for_status()
                response_data = response.json()
                
                # 解析响应
                answer = response_data["choices"][0]["message"]["content"]
            
            # print("已收到API的响应")
            return answer
            
        except Exception as e:
            error_msg = f"生成答案时出错: {str(e)}"
            print(error_msg)
            return error_msg
    
    def answer_with_sources(self, query: str, vector_store, top_k=5) -> Dict[str, Any]:
        """
        使用检索到的文档回答问题并显示来源
        """
        print(f"正在回答问题: {query}")
        
        # 检索相关文档
        print(f"正在检索排名前 {top_k} 的相关文档...")
        retrieved_docs = vector_store.similarity_search(query, top_k=top_k)
        
        # 生成答案
        answer = self.generate_answer(query, retrieved_docs)
        
        # 格式化响应
        response = {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "similarity": doc["similarity"]
                }
                for doc in retrieved_docs
            ]
        }
        
        return response