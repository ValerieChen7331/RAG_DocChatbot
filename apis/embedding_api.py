from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os


class EmbeddingAPI:
    @staticmethod
    def get_embedding_function(mode, embedding):
        """選擇內部或外部 LLM 模型來獲取 embeddings"""
        if mode == '內部LLM':
            # embedding = 'text-embedding-ada-002'
            # return EmbeddingAPI._get_external_embeddings(embedding)
            return EmbeddingAPI._get_internal_embeddings(embedding)
        else:

            return EmbeddingAPI._get_external_embeddings(embedding)

    @staticmethod
    def _get_internal_embeddings(embedding):
        """獲取內部 LLM 模型的 embeddings"""
        # 定義內部可用的 embedding 模型與 base_url
        embedding_models = {
            "llama3": "http://10.5.61.81:11435",
            "bge-m3": "http://10.5.61.81:11437"
        }
        # 檢查模型名稱是否有效
        base_url = embedding_models.get(embedding)
        if not base_url:
            raise ValueError(f"無效的內部 embeddings 模型名稱：{embedding}")

        # 建立並返回 OllamaEmbeddings 實例
        embeddings = OllamaEmbeddings(base_url=base_url, model="bge-m3:latest")
        return embeddings

    @staticmethod
    def _get_external_embeddings(embedding):
        """獲取外部 Azure 模型的 embeddings"""
        # 加载 .env 文件中的环境变量
        load_dotenv()

        # 从环境变量中获取 API Key、Endpoint 和 API 版本
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        embedding_api_version = os.getenv("Embedding_API_VERSION")

        # 使用 Azure OpenAI API 來建立 embeddings
        embeddings = AzureOpenAIEmbeddings(
            model=embedding,
            azure_endpoint=api_base,
            api_key=api_key,
            openai_api_version=embedding_api_version,
            # dimensions: Optional[int] = None  # 可選擇指定新 text-embedding-3 模型的維度
        )
        return embeddings
