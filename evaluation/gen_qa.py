# qa_generator.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from tqdm import tqdm
from apis.llm_api import LLMAPI
from langchain.prompts import PromptTemplate
import time



class AnswerGenerator:
    def __init__(self, llm_option: str):
        # 初始化：指定要使用的 LLM 模型選項
        self.llm_option = llm_option
        self.llm = LLMAPI.get_llm("內部LLM", llm_option)  # 從 LLMAPI 取得對應的模型實例

        # 設定提示詞模板，用於構建每一筆問答的 prompt
        self.template = PromptTemplate(
            input_variables=["content", "query"],
            template="""
            【回覆格式要求】
            您是專業的問答助手，請根據下方檢索到的內容來回答問題。
            1. 針對問題判斷需參考哪些文件內容，不必強求使用所有文件。
            2. 回答時，請先引述所用文件的原文與頁碼，再進行說明與回覆。
            3. 若檢索內容中無法得出明確答案，請誠實回答「我不知道」，不得推測或提供無根據的回覆。

            【參考文件】
            {content}

            【問題】
            {query}
            """
        )

    def generate_answers(self, qa_file_path: str, output_file_path: str):
        # 從 QA CSV 檔案讀取資料（須包含 Question 和 DocContents 欄位）
        df = pd.read_csv(qa_file_path)
        answers = []
        print(f"🤖 啟動問答生成：{self.llm_option}")

        # 使用 tqdm 顯示進度條，單位為「題」，欄寬設為 80，顯示當前模型名稱
        # 使用簡潔版 tqdm 進度條設定（固定欄寬、顯示單位）
        for idx in tqdm(range(len(df)), desc="處理中", unit="題", leave=True, ncols=100):
            # 依序取得問題與文件內容
            question = df.loc[idx, 'Question']
            content = df.loc[idx, 'DocContents']
            try:
                # 將內容與問題帶入提示詞模板，呼叫模型生成回答
                prompt = self.template.format(content=content, query=question)
                answer = self.llm.invoke(prompt)
            except Exception as e:
                # 若發生錯誤則紀錄錯誤訊息
                print(f"⚠️ 回答生成失敗：{e}")
                answer = f"錯誤: {e}"
            answers.append(answer)
            time.sleep(3)

        # 將模型回應結果新增至新欄位（依照模型命名）
        df[f'Answer_{self.llm_option}'] = answers

        # 建立輸出資料夾並儲存回應結果為 CSV
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"✅ 結果已儲存：{output_file_path}")
