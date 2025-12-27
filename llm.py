import ollama
import os


class LLMClassifier:
    def __init__(self, model_name="qwen2.5:3b"):
        self.model_name = model_name

    def classify(self, file_path, text_content, topics):
        """
        1. 读取文本摘要
        2. 调用 Ollama 判别分类
        3. 移动文件
        """
        # 截取前 2000 个字符作为摘要，避免超出 Context Window
        abstract = text_content[:2000]
        topic_str = ", ".join(topics)

        prompt = (
            f"你是一个专业的文献管理员。请根据以下论文内容摘要，将其归类到以下类别之一：[{topic_str}]。\n"
            f"请直接返回最匹配的一个类别名称，不要包含任何其他解释或标点符号。\n\n"
            f"论文内容：{abstract}..."
        )

        try:
            print(f"正在使用Qwen2.5-3B分析论文: {os.path.basename(file_path)}...")
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt},
            ])
            category = response['message']['content'].strip()

            # 清理一下返回值，防止模型废话（比如 "类别是：CV" -> "CV"）
            for t in topics:
                if t in category:
                    category = t
                    break

            # 如果 AI 返回的类别不在列表里，归入 "Uncategorized"
            if category not in topics:
                category = "Uncategorized"

            return category

        except Exception as e:
            print(f"❌ 分类失败: {e}")
            return None