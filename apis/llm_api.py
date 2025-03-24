from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
from langchain_community.llms import Ollama


class LLMAPI:
    @staticmethod
    def get_llm(mode, llm_option):
        """
        根據傳入的模式 (mode) 及模型選項 (llm_option)，
        決定要呼叫內部 LLM 還是外部 Azure LLM。

        :param mode: 字串，'內部LLM' 或其他 (預設為外部 LLM)
        :param llm_option: 模型名稱選項
        :return: LLM 實例
        """
        if mode == '內部LLM':
            return LLMAPI._get_internal_llm(llm_option)
        else:
            return LLMAPI._get_external_llm(llm_option)

    @staticmethod
    def _get_internal_llm(llm_option):
        """
        建立並回傳內部 LLM (透過 Ollama 伺服器)

        :param llm_option: 模型選項名稱 (key)
        :return: Ollama LLM 實例
        """
        api_base = "http://10.5.61.81:11436"  # 內部 Ollama API 位址

        # 定義內部支援的模型清單 (字典)
        llm_model_names = {
            "Taiwan-llama3-f16": "cwchang/llama-3-taiwan-8b-instruct-dpo:f16",
            "Taiwan-llama3-8b": "SimonPu/llama-3-taiwan-8b-instruct-dpo",
            "Gemma2": "gemma2:latest",
            "Gemma2:27b": "gemma2:27b-instruct-q5_0",
            "Taide-llama3-8b-f16": "jcai/llama3-taide-lx-8b-chat-alpha1:f16"
        }

        # 根據傳入 llm_option 找對應模型
        model = llm_model_names.get(llm_option)
        if not model:
            raise ValueError(f"無效的內部模型選項：{llm_option}")

        # 回傳 Ollama LLM 實例，使用指定模型
        llm = Ollama(base_url=api_base, model=model)
        return llm

    @staticmethod
    def _get_external_llm(llm_option):
        """
        建立並回傳 Azure OpenAI 的 LLM 實例 (外部)

        :param llm_option: Azure 上已部署的模型名稱 (deployment name)
        :return: AzureChatOpenAI LLM 實例
        """
        deployment_name = llm_option
        load_dotenv()  # 載入 .env 檔案環境變數

        # 從環境變數取得 Azure OpenAI API 設定
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        # 若有缺少必要的變數，丟出錯誤
        if not all([api_key, api_base, api_version]):
            raise ValueError("缺少 API Key、Endpoint 或 API Version")

        # 建立 AzureChatOpenAI LLM 實例
        llm = AzureChatOpenAI(
            openai_api_key=api_key,
            azure_endpoint=api_base,
            api_version=api_version,
            deployment_name=deployment_name
        )
        return llm


if __name__ == "__main__":
    # 範例：使用內部 LLM 測試
    llm = LLMAPI.get_llm("內部LLM", "Taiwan-llama3-f16")
    # 使用 Chat 格式 (list of dict) 呼叫模型
    result = llm.invoke([
        {"role": "user", "content": "我要如何設定自動寄送的警報信？"}
    ])
    # 將結果印出
    print(result)
