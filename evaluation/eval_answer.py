# answer_evaluator.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import sqlite3
import json
from tqdm import tqdm
from apis.llm_api import LLMAPI
import time
from datetime import datetime

class AnswerEvaluator:
    def __init__(self, correct_file: str, llm_input_file: str, output_file: str,
                 mode: str, llm_option: str, evaluator_llm: str = "Gemma3_27b", evaluation_attempts: int = 3):
        # 初始化評估器，設定所需參數
        self.correct_file = correct_file  # 正確答案來源檔案
        self.llm_input_file = llm_input_file  # LLM 回答來源檔案
        self.output_file = output_file  # 評估結果儲存位置
        self.llm_option = llm_option  # 被評估的模型名稱
        self.evaluation_attempts = evaluation_attempts  # 評估次數

        # 初始化評估使用的 LLM（可與產生回答的模型不同）
        self.llm = LLMAPI.get_llm(mode, evaluator_llm)

        # 評估使用的提示詞範本
        self.prompt_template = """
        請比較以下內容，並回答「*t」(代表回答正確)或「*f」(代表回答錯誤)：
        1. 檢查「實際回應」中，是否包含「正確答案」中的必要資訊和核心內容。
        2. 檢查連結金額數字是否正確。
        3. 檢查連結 url 是否正確。
        4. 不拘泥於文字形式，確保意思一致。若「實際回應」中有思考過程，可忽略不列入評估，以結論為主。
        5. 請列出思考過程，並給出結論，最後輸出: final_answer:「*t」或「*f」。
        ---
        (1.) 問題: {query}
        (2.) 正確答案：{expected_response}
        (3.) 實際回應：{generated_response}
        """

    def evaluate_answers(self):
        # 讀取正確答案與 LLM 回答的資料表
        df_llm = pd.read_csv(self.llm_input_file)
        df_correct = pd.read_csv(self.correct_file)

        # 合併兩表格，以 ID 為主鍵補上正確答案欄位
        merged_df = pd.merge(df_llm, df_correct[['ID', 'Answer']], on='ID', how='left')

        # 檢查是否存在目標回答欄位（以模型名稱命名）
        test_column = f"Answer_{self.llm_option}"
        if test_column not in merged_df.columns:
            raise ValueError(f"❌ 欄位不存在：{test_column}，請確認 llm_input_file 是否正確。")

        # 多輪評估，每輪評估所有問題
        evaluations = []
        for attempt in range(self.evaluation_attempts):
            scores = []
            for idx, row in tqdm(
                    merged_df.iterrows(),
                    desc=f"第 {attempt + 1} 輪評估中：{self.llm_option}",
                    unit="題",
                    leave=True,
                    dynamic_ncols=True
            ):
                # 執行單筆回答的評估
                scores.append(
                    self._evaluate_single_response(row['Question'], row['Answer'], row[test_column])
                )
                time.sleep(3)
            evaluations.append(scores)
            # 跑多輪，中間休息
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"✅ 完成模型回答評估：{self.llm_option}（時間：{current_time}），暫停 5 分鐘避免限流")
            time.sleep(250)


        # 計算平均得分與是否正確
        avg_scores = [sum(score_set)/len(score_set) for score_set in zip(*evaluations)]
        merged_df['SimilarityScore'] = avg_scores
        merged_df['SimilarityBoolean'] = [score > 0.5 for score in avg_scores]

        # 顯示評估總結資訊
        correct_count = sum(merged_df['SimilarityBoolean'])
        total_count = len(merged_df)
        average_score = sum(avg_scores) / total_count if total_count > 0 else 0
        print("📊 評估總結：")
        print(f"✔️ correct_similarity 正確回答數量：{correct_count} / {total_count}")
        print(f"⭐ 平均相似度分數：{average_score:.2f}")
        # 儲存統計結果
        summary_path = self.output_file.replace(".csv", "_summary.csv")
        df = pd.DataFrame([{
            "CorrectCount": correct_count,
            "TotalCount": total_count,
            "AverageScore": average_score
        }])
        df.to_csv(summary_path, index=False, encoding='utf-8-sig')

        # 儲存結果為 CSV
        merged_df.to_csv(self.output_file, index=False, encoding='utf-8-sig')
        # 儲存至 SQLite
        self._save_to_db(merged_df)

        # 只跑一輪
        # current_time = datetime.now().strftime('%H:%M:%S')
        # print(f"✅ 完成模型回答評估：{self.llm_option}（時間：{current_time}），暫停 5 分鐘避免限流")
        # time.sleep(300)

    def _evaluate_single_response(self, query, expected_response, generated_response) -> int:
        # 將輸入代入提示詞模板
        prompt = self.prompt_template.format(
            query=query,
            expected_response=expected_response,
            generated_response=generated_response
        )
        try:
            # 呼叫評審 LLM 取得回應結果
            result = self.llm.invoke(prompt).strip().lower()
            # print("🧪 LLM 回傳原始內容：", result)
            final = result.split('final_answer:')[-1].strip()
            return 1 if '*t' in final else 0
        except Exception as e:
            print("❌ 發生錯誤：", e)
            return 0

    def _save_to_db(self, df):
        # 儲存評估結果至 SQLite 資料庫（欄位與 merged_df 完全一致）
        db_path = self.output_file.replace('.csv', '.db')
        with sqlite3.connect(db_path) as conn:
            # 若有 Docs 欄位，先轉為 JSON 字串格式
            if 'Docs' in df.columns:
                df['Docs'] = df['Docs'].apply(
                    lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x
                )

            # 將欄位 ID 改名為 question_id 以避免與主鍵 id 衝突
            if 'ID' in df.columns:
                df = df.rename(columns={'ID': 'question_id'})

            # 自動寫入 DataFrame（若資料表不存在會自動建立）
            try:
                df.to_sql('evaluations', conn, if_exists='append', index=False)
                print(f"📝 已寫入 SQLite：{db_path}")
            except Exception as e:
                print("❌ 寫入 SQLite 發生錯誤：", e)