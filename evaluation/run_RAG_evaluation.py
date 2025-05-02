# run_RAG_evaluation.py
import os
import sys
import sqlite3
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from eval_answer import AnswerEvaluator
from apis.llm_api import LLMAPI
import pandas as pd

test_num = 3

def run_step1_data():
    # 步驟二：從 Guest.db 中撈出 ID, user_query, ai_response 存成 csv，更名為 ID, Question, Answer_{llm_option}
    userDB_path = "data/user/Guest/Guest.db"
    output_dir = "mockdata"

    # 確保輸出資料夾存在
    os.makedirs(output_dir, exist_ok=True)

    # 連線資料庫
    conn = sqlite3.connect(userDB_path)
    cursor = conn.cursor()

    # 撈出所有有 ai_response 的紀錄
    cursor.execute("""
        SELECT id, llm_option, user_query, ai_response
        FROM chat_history
        WHERE ai_response IS NOT NULL AND TRIM(ai_response) != ''
    """)
    rows = cursor.fetchall()

    if not rows:
        print("⚠️ 無資料可匯出，請確認 chat_history 資料是否存在 AI 回答")
        return

    # 根據 llm_option 分組匯出
    data_by_model = {}
    for row in rows:
        id, llm_option, user_query, ai_response = row
        if llm_option not in data_by_model:
            data_by_model[llm_option] = []
        data_by_model[llm_option].append([id, user_query, ai_response])

    for llm_option, data in data_by_model.items():
        # 修改欄位名稱為 Answer_{llm_option}
        df = pd.DataFrame(data, columns=["ID", "Question", f"Answer_{llm_option}"])
        model_dir = os.path.join(output_dir, llm_option)
        os.makedirs(model_dir, exist_ok=True)
        output_path = os.path.join(model_dir, f"test{test_num}_{llm_option}_response.csv")
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已匯出模型回答：{output_path}")

    conn.close()

def run_step2_evaluate_answers():
    evaluator_llm = "Gemma3_27b"
    llm_option = "Gemma3_27b"
    llm_input_file = f"mockdata/{llm_option}/test{test_num}_{llm_option}_response.csv"
    output_path = f"mockdata/{llm_option}/test{test_num}_{llm_option}_evaluate_{evaluator_llm}.csv"
    if os.path.exists(llm_input_file):
        print(f"🔍 開始評估模型回答：{llm_option}")
        AnswerEvaluator(
            correct_file="mockdata/QAData_1819.csv",
            llm_input_file=llm_input_file,
            output_file=output_path,
            mode="內部LLM",
            llm_option=llm_option,
            evaluator_llm=evaluator_llm,
            evaluation_attempts=3
        ).evaluate_answers()
    else:
        print(f"❌ 找不到檔案，略過評估：{llm_input_file}")

if __name__ == "__main__":
    # Step 2：針對每個模型產生回答
    run_step1_data()

    # Step 3：對產生的回答進行模型自動評估
    run_step2_evaluate_answers()
