import os
import fitz  # PyMuPDF
import re
from tqdm import tqdm
import pandas as pd

class PDFExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.filename = os.path.basename(pdf_path)
        self.doc = fitz.open(pdf_path)
        self.num_pages = len(self.doc)
        print(f"加载PDF: {self.filename}，共 {self.num_pages} 页")
    
    def extract_text_with_metadata(self, chunk_size=500, overlap=50):
        """
        从PDF中提取文本，并附带页码和段落元数据。
        返回一个包含文本和元数据的块列表。
        """
        all_chunks = []
        
        for page_num in tqdm(range(self.num_pages), desc=f"正在处理 {self.filename}"):
            page = self.doc[page_num]
            
            # 1. 首先尝试通过布局分析提取结构化文本
            paragraphs = self._extract_paragraphs_from_page(page)
            
            for para_num, paragraph in enumerate(paragraphs):
                # 清理文本：移除多余的空白，但保留必要的结构
                cleaned_text = re.sub(r'\s{2,}', ' ', paragraph).strip()
                
                if not cleaned_text or len(cleaned_text) < 10:  # 跳过太短的段落
                    continue
                
                # 添加元数据
                metadata = {
                    "source": self.filename,
                    "page_number": page_num + 1,  # 使用从1开始的页码更符合用户习惯
                    "paragraph_number": para_num + 1,
                    "total_pages": self.num_pages
                }
                
                # 对长段落进行分块
                if len(cleaned_text) <= chunk_size:
                    all_chunks.append({"text": cleaned_text, "metadata": metadata})
                else:
                    # 对长段落进行重叠分块
                    for i in range(0, len(cleaned_text), chunk_size - overlap):
                        chunk_text = cleaned_text[i:i + chunk_size]
                        if len(chunk_text) < 50:  # 跳过过小的块
                            continue
                        
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_number"] = i // (chunk_size - overlap) + 1
                        all_chunks.append({"text": chunk_text, "metadata": chunk_metadata})
        
        print(f"从 {self.filename} 中提取了 {len(all_chunks)} 个文本块")
        return all_chunks
    
    def _extract_paragraphs_from_page(self, page):
        """从页面中提取段落，考虑文本结构和布局"""
        # 尝试使用块提取文本，这保留了布局信息
        blocks = page.get_text("dict")["blocks"]
        
        if not blocks:
            # 回退到基本文本提取
            return self._extract_paragraphs_from_text(page.get_text())
        
        paragraphs = []
        current_paragraph = []
        last_block_type = None
        last_y1 = 0
        
        # 遍历文本块
        for block in blocks:
            if "lines" not in block:
                continue
                
            # 识别段落的启发式规则
            block_type = block.get("type", 0)
            y0 = block.get("bbox", [0, 0, 0, 0])[1]  # y坐标
            
            # 处理文本行
            for line in block["lines"]:
                if "spans" not in line:
                    continue
                
                line_text = ""
                for span in line["spans"]:
                    if "text" in span and span["text"].strip():
                        line_text += span["text"] + " "
                
                line_text = line_text.strip()
                if not line_text:
                    continue
                
                # 判断这是否是新段落的开始
                is_new_paragraph = False
                
                # 基于缩进检测段落
                if line_text.startswith("    ") or line_text.startswith("\t"):
                    is_new_paragraph = True
                
                # 基于垂直间距检测段落
                elif abs(y0 - last_y1) > 15:  # 行间距阈值，可根据PDF调整
                    is_new_paragraph = True
                    
                # 基于上一块类型检测段落
                elif last_block_type is not None and last_block_type != block_type:
                    is_new_paragraph = True
                
                if is_new_paragraph and current_paragraph:
                    # 完成当前段落
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
                
                current_paragraph.append(line_text)
                last_y1 = line.get("bbox", [0, 0, 0, 0])[3]  # 更新最后一行的y1
            
            last_block_type = block_type
        
        # 添加最后一个段落
        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))
        
        # 合并过短的段落（可能是因为PDF格式导致的不正确分段）
        merged_paragraphs = []
        temp_paragraph = ""
        
        for p in paragraphs:
            # 如果当前段落不到30个字符并且不是以句号、问号或感叹号结束，可能是不完整的段落
            if len(p) < 30 and not re.search(r'[.。?？!！]$', p):
                temp_paragraph += " " + p
            else:
                if temp_paragraph:
                    temp_paragraph += " " + p
                    merged_paragraphs.append(temp_paragraph.strip())
                    temp_paragraph = ""
                else:
                    merged_paragraphs.append(p)
        
        if temp_paragraph:
            merged_paragraphs.append(temp_paragraph.strip())
            
        # 如果提取失败或结果太少，回退到基本文本提取
        if not merged_paragraphs:
            return self._extract_paragraphs_from_text(page.get_text())
            
        return merged_paragraphs
    
    def _extract_paragraphs_from_text(self, text):
        """从纯文本中提取段落，用于回退"""
        # 更强的段落分割规则
        raw_paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = []
        
        for raw_para in raw_paragraphs:
            if not raw_para.strip():
                continue
                
            # 处理有换行符的段落
            lines = raw_para.split('\n')
            current_para = []
            current_line = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    # 空行表示段落分隔
                    if current_line:
                        current_para.append(current_line)
                        current_line = ""
                    continue
                
                # 判断是否应该将这一行作为新段落的开始
                is_new_paragraph = False
                
                # 检查是否是新段落的标志（缩进、编号等）
                if re.match(r'^(\d+\.|\•|\*|\-|\t|    )', line):
                    is_new_paragraph = True
                
                # 如果上一行结束有完整句号，这可能是新段落
                elif current_line and re.search(r'[.。?？!！]$', current_line):
                    is_new_paragraph = True
                
                # 当前行太短，可能与下一行连续
                elif len(line) < 30 and current_line and not re.search(r'[.。?？!！]$', current_line):
                    is_new_paragraph = False
                
                if is_new_paragraph and current_line:
                    current_para.append(current_line)
                    current_line = line
                else:
                    if current_line:
                        current_line += " " + line
                    else:
                        current_line = line
            
            # 添加最后一行
            if current_line:
                current_para.append(current_line)
            
            if current_para:
                paragraphs.extend(current_para)
        
        return paragraphs
    
    def get_document_metadata(self):
        """返回文档的基本元数据"""
        return {
            "filename": self.filename,
            "total_pages": self.num_pages,
            "author": self.doc.metadata.get("author", "未知"),
            "title": self.doc.metadata.get("title", os.path.splitext(self.filename)[0])
        }
    
    def save_extracted_text(self, output_dir):
        """将提取的文本保存为CSV以便检查/调试"""
        chunks = self.extract_text_with_metadata()
        
        # 为DataFrame创建扁平结构
        rows = []
        for chunk in chunks:
            row = {
                "text": chunk["text"],
                **chunk["metadata"]  # 解包元数据
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为CSV
        output_path = os.path.join(output_dir, f"{os.path.splitext(self.filename)[0]}_extracted.csv")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"已将提取的文本保存到 {output_path}")
        
        return chunks
    
    def close(self):
        """关闭文档以释放资源"""
        self.doc.close()

def extract_all_pdfs(pdf_dir, output_dir="data/extracted"):
    """从目录中提取所有PDF文本并返回合并的块"""
    all_chunks = []
    
    # 获取目录中的所有PDF文件
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        extractor = PDFExtractor(pdf_path)
        
        try:
            # 提取并保存为CSV
            chunks = extractor.save_extracted_text(output_dir)
            all_chunks.extend(chunks)
        finally:
            extractor.close()
    
    print(f"所有PDF共提取了 {len(all_chunks)} 个文本块")
    return all_chunks