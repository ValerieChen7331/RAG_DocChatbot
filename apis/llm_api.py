from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
from langchain_community.llms import Ollama

class LLMAPI:
    """
    提供 LLM（大型語言模型）的統一初始化介面。
    根據使用者選擇的模式（內部或外部），取得對應模型。
    """

    # 支援的內部 LLM 模型對應表（顯示名稱 → 實際模型 ID）
    llm_model_names = {
        "Gemma3_27b": "gemma3:27b",
        "Deepseek_14b_QwenDistill": "deepseek-r1:14b-qwen-distill-q4_K_M",
        "Mistral_7b_Instruct": "mistral:7b-instruct",
        "Gemma2_27b": "gemma2:27b-instruct-q5_0",
        "Gemma3_4b": "gemma3:4b",
        "Deepseek_7b": "deepseek-r1:7b",
        "Phi4_14b": "phi4:14b",
        "LLaMA3_2_Latest": "llama3.2:latest",
        "Taiwan_LLaMA3_8b_Instruct": "cwchang/llama-3-taiwan-8b-instruct:f16",
        "QWQ_32b": "qwq:32b",
    }
    # "Gemma3_12b": "gemma3:12b",

    @staticmethod
    def get_llm(mode: str, llm_option: str):
        """
        根據模式（內部/外部）取得對應的 LLM 實例。
        """
        if mode == '內部LLM':
            return LLMAPI._get_internal_llm(llm_option)
        else:
            return LLMAPI._get_external_llm(llm_option)

    @staticmethod
    def _get_internal_llm(llm_option: str):
        """
        回傳本地部署（Ollama）架構的 LLM 實例。
        預設使用 API 位置為本地/內網部署。
        """
        api_base = "http://10.5.61.81:11437"

        # 從模型對應表中取得實際模型 ID
        model = LLMAPI.llm_model_names.get(llm_option)
        if not model:
            raise ValueError(f"❌ 無效的內部模型選項：{llm_option}")

        return Ollama(base_url=api_base, model=model)

    @staticmethod
    def _get_external_llm(llm_option: str):
        """
        回傳 Azure OpenAI 平台的 LLM 實例。
        會從 .env 讀取金鑰與設定。
        """
        # 載入 .env 環境變數
        load_dotenv()

        # 從環境變數中取得必要設定
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        # 驗證所有設定是否存在
        if not all([api_key, api_base, api_version]):
            raise ValueError("❌ 缺少 Azure OpenAI API 設定（Key、Endpoint 或 Version）")

        # 回傳 AzureChatOpenAI 實例（使用 langchain 封裝）
        return AzureChatOpenAI(
            openai_api_key=api_key,
            azure_endpoint=api_base,
            api_version=api_version,
            deployment_name=llm_option
        )
