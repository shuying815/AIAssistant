# paper_manager.py
from unicodedata import category

from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import os
import shutil
from glob import glob
import numpy as np
import fitz
from llm import LLMClassifier

class PaperManager:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
      #  self.client = chromadb.Client()
        self.client = chromadb.PersistentClient(path="/Users/shuying/Desktop/AI助手/database")
        self.collection = self.client.get_or_create_collection(name="papers")

        self.categories = ['Computer Vision', 'Natural Language Processing', 'Reinforcement Learning']  # 定义主题分类
        self.category_embeddings = self._generate_category_embeddings(self.categories)
        self.paper_dir = "/Users/shuying/Desktop/AI助手/paper"

        self.classifier = LLMClassifier()

    def _generate_category_embeddings(self, categories):
        # 生成每个类别的嵌入向量（可以根据类别名称生成嵌入）
        category_embeddings = {}
        for category in categories:
            category_embeddings[category] = self.model.encode([category])[0]
        return category_embeddings

    def add_paper(self, paper_path, topics = None):

        if os.path.isdir(paper_path):
            # 如果是文件夹路径，批量处理该文件夹中的所有PDF文件
            self.batch_process(paper_path, topics)
        elif os.path.isfile(paper_path) and paper_path.endswith('.pdf'):
            # 如果是单个PDF文件，处理该文件
            self._process_single_paper(paper_path, topics)
        else:
            print(f"===>{paper_path} 不是有效的文件或文件夹路径")
            exit()

    def search_paper(self, query):
        # 检查数据库中的文献数量
        print('===>数据库中论文总数: ', self.collection.count())

        query_embedding = self.model.encode([query])
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=1  # 返回最相关的3篇文献
        )

        #file_paths = [result[0]['source'] for result in results['metadatas']]
        return results

    def extract_text_from_pdf(self, pdf_path):
        """提取PDF文本，返回 (full_text, segments_list)"""
        doc = fitz.open(pdf_path)
        full_text = ""
        segments = []
        metadatas = []

        for page_num, page in enumerate(doc):
            text = page.get_text()
            if len(text) < 50: continue  # 跳过空白页或字数太少的页

            full_text += text + "\n"

            # 简单按页切分，进阶可以按段落切分
            segments.append(text)
            metadatas.append({
                "path": pdf_path,
                "page": page_num + 1,
                "filename": pdf_path.split("/")[-1]
            })

        return full_text, segments, metadatas

    def get_embedding(self, text):
        """获取单个文本的向量"""
        # 返回 list 格式以适配 ChromaDB
        return self.model.encode(text).tolist()

    def get_embeddings(self, texts):
        """批量获取向量"""
        return self.model.encode(texts).tolist()

    def add_paper_segments(self, file_path, segments, embeddings, metadatas):
        """批量存储论文片段"""
        # 生成唯一ID: 文件名_页码_索引
        ids = [f"{os.path.basename(file_path)}_{m['page']}_{i}" for i, m in enumerate(metadatas)]

        self.collection.add(
            documents=segments,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"已存入 {len(segments)} 个片段: {os.path.basename(file_path)}")


    def _process_single_paper(self, paper_path, topics=None):

        # 读取文件并转换为文本 (假设是 PDF)
        full_text, segments, metadatas = self.extract_text_from_pdf(paper_path)
        embeddings = self.get_embeddings(segments)

        # 创建目标文件夹路径
        if topics == None:
            topics = self.categories

        for t in topics:
            folder_path = os.path.join(self.paper_dir, t)
            os.makedirs(folder_path, exist_ok=True)

        category = self.classifier.classify(paper_path, full_text, topics)

        # 移动论文到对应的文件夹
        target_path = os.path.join(self.paper_dir, category, os.path.basename(paper_path))
        shutil.move(paper_path, target_path)
        print(f"===>论文 '{paper_path}' 被归类到 '{category}', 新路径为 '{target_path}'.")

        # 存入数据库 (用于搜索)
        self.add_paper_segments(target_path, segments, embeddings, metadatas)


    def batch_process(self, folder_path, topics = None):
        # 扫描文件夹中的所有PDF文件
        pdf_files = glob(os.path.join(folder_path, "*.pdf"))
        for pdf_file in pdf_files:
            print(f"===>Processing paper: {pdf_file}")
            self._process_single_paper(pdf_file, topics)
