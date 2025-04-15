import os
import sys
import json
from dotenv import load_dotenv

# 将当前目录添加到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模块
from pdf_extractor import extract_all_pdfs
from vector_store import VectorStore
from qa_system import QASystem

def setup_knowledge_base(pdf_dir, force_rebuild=True):
    """
    从PDF文件建立知识库
    """
    print("\n" + "="*50)
    print("智能数据工程课程知识库构建系统")
    print("="*50)
    
    # 从PDF提取文本
    print("\n[1/3] 从PDF提取文本...")
    
    # 检查是否已经完成提取
    extracted_dir = os.path.join("data", "extracted")
    if not force_rebuild and os.path.exists(extracted_dir) and len(os.listdir(extracted_dir)) > 0:
        print("检测到已提取的文本数据。跳过提取步骤...")
    else:
        print(f"从目录 {pdf_dir} 提取PDF文本...")
        all_chunks = extract_all_pdfs(pdf_dir, output_dir=extracted_dir)
        print(f"文本提取完成。总共提取了 {len(all_chunks)} 个文本块")
    
    # 初始化向量存储
    print("\n[2/3] 构建向量存储...")
    vector_store = VectorStore(persist_directory="data/vector_db")
    
    if force_rebuild:
        vector_store.reset_collection()

    # 检查向量存储是否已填充
    if not force_rebuild and vector_store.get_stats()["document_count"] > 0:
        print("检测到已构建的向量存储。跳过向量构建步骤...")
    else:
        # 加载已提取的文本块
        all_chunks = []
        for filename in os.listdir(extracted_dir):
            if filename.endswith("_extracted.csv"):
                print(f"处理已提取文件: {filename}")
                
                # 从CSV加载文本块（在提取过程中创建）
                import pandas as pd
                df = pd.read_csv(os.path.join(extracted_dir, filename), encoding='utf-8')
                
                for _, row in df.iterrows():
                    chunk = {
                        "text": row["text"],
                        "metadata": {
                            "source": row["source"],
                            "page_number": row["page_number"],
                            "paragraph_number": row["paragraph_number"],
                            "total_pages": row["total_pages"]
                        }
                    }
                    all_chunks.append(chunk)
        
        # 将文本块添加到向量存储
        vector_store.add_documents(all_chunks)
        
        # 保存向量存储内容以便检查
        vector_store.save_content_for_inspection()
    
    # 初始化问答系统
    print("\n[3/3] 初始化问答系统...")
    qa_system = QASystem()
    
    print("\n知识库构建完成！")
    return vector_store, qa_system

def interactive_qa(vector_store, qa_system):
    """
    交互式问答会话
    """
    print("\n" + "="*50)
    print("智能数据工程课程知识问答系统")
    print("="*50)
    print("输入问题与课程内容进行交互，输入 'exit' 或 'quit' 退出")
    
    while True:
        # 从用户获取问题
        query = input("\n请输入问题: ")
        
        # 检查用户是否想要退出
        if query.lower() in ["exit", "quit", "退出"]:
            print("谢谢使用，再见！")
            break
        
        # 回答问题
        response = qa_system.answer_with_sources(query, vector_store, top_k=5)
        
        # 打印答案
        print("\n回答:")
        print(response["answer"])

def main():
    """
    运行知识库和问答系统的主函数
    """
    # 加载环境变量
    load_dotenv()
    
    # 获取PDF目录的路径
    pdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Resource")
    
    # 检查目录是否存在
    if not os.path.exists(pdf_dir):
        print(f"错误: PDF目录 '{pdf_dir}' 不存在!")
        return
    
    # 检查命令行参数
    force_rebuild = "--force-rebuild" in sys.argv
    
    # 设置知识库
    vector_store, qa_system = setup_knowledge_base(pdf_dir, force_rebuild=force_rebuild)
    
    # 启动交互式问答
    interactive_qa(vector_store, qa_system)

if __name__ == "__main__":
    main()