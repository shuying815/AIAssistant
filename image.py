# image_manager.py
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import chromadb
import uuid

class ImageManager:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.client = chromadb.PersistentClient(path="/Users/shuying/Desktop/AI助手/database")
        self.collection = self.client.get_or_create_collection(name="images")

    def add_image(self, image_path):
        # 读取图像
        image = Image.open(image_path)
        inputs = self.processor(images=[image], return_tensors="pt", padding=True)

        # 获取图像和文本的嵌入
        with torch.no_grad():
            image_embedding = self.model.get_image_features(**inputs).squeeze()
        image_id = str(uuid.uuid4())
        # 将图像描述和嵌入存储到数据库
        self.collection.add(
            metadatas=[{"source": image_path}],
            embeddings=[image_embedding.numpy()],
            ids = [image_id]
        )

    def search_image(self, query):
        query_input = self.processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            query_embedding = self.model.get_text_features(**query_input)

        print('===>数据库中图像数: ', self.collection.count())

        # 在图像数据库中查询最相似的图像
        image_results = self.collection.query(
            query_embeddings=query_embedding.numpy(),
            n_results=1  # 返回最相关的3张图像
        )
        file_paths = [result[0]['source'] for result in image_results['metadatas']]
        return file_paths

