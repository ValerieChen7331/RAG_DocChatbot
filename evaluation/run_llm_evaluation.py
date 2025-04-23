# run_llm_evaluation.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from export_data import RAGDataExporter
from gen_qa import AnswerGenerator
from eval_answer import AnswerEvaluator
from apis.llm_api import LLMAPI

test_num = 1

def run_step1_export():
    # 步驟一：從 SQLite 資料庫中匯出所有 RAG 過程中的 Chunks，產出 QAData_retrievedContent.csv (不一定有GoldenChunk)
    db_path = "data/user/Guest/Guest.db"
    qa_retrieved_path = "mockdata/QAData_retrievedContent.csv"
    RAGDataExporter(db_path).export_to_csv(qa_retrieved_path)
    # 手動加入 GoldenChunk (未寫程式)
    # golden_chunk_path = "mockdata/QAData_goldenChunk.csv"


def run_step2_generate_answers():
    # 步驟二：根據 QA CSV，針對每個模型生成回答並輸出到 mockdata/{模型名}/{模型名}_results.csv
    qa_csv_path = "mockdata/QAData_goldenChunk.csv"
    for llm_option in LLMAPI.llm_model_names.keys():
        output_path = f"mockdata/{llm_option}/test{test_num}_{llm_option}_results.csv"
        AnswerGenerator(llm_option).generate_answers(qa_csv_path, output_path)
        print(f"✅ 完成模型回答生成：{llm_option}，暫停 5 分鐘避免限流")
        _countdown_sleep(300)
def run_step3_evaluate_answers():
    # 步驟三：讀取 QA 標準答案檔與模型產生的回答，並執行評估，輸出評估結果與分數
    evaluator_llm = "Gemma3_27b"
    for llm_option in LLMAPI.llm_model_names.keys():
        llm_input_file = f"mockdata/{llm_option}/test{test_num}_{llm_option}_results.csv"
        output_path = f"mockdata/{llm_option}/test{test_num}_{llm_option}_evaluate_{evaluator_llm}.csv"
        if os.path.exists(llm_input_file):
            print(f"🔍 開始評估模型回答：{llm_option}")
            AnswerEvaluator(
                correct_file="mockdata/QAData_1819.csv",    # 正確答案 CSV 檔案（含欄位 ID, Question, Answer）
                llm_input_file=llm_input_file,              # 模型產生的回答檔案（含 Answer_{llm_option} 欄位）
                output_file=output_path,                    # 輸出評估結果的檔案路徑
                mode="內部LLM",                              # 使用內部模型做為評審
                llm_option=llm_option,                      # 當前要評估的模型名稱
                evaluator_llm=evaluator_llm,                    # 擔任評審的模型（如 GPT-4o 或 Gemma）
                evaluation_attempts=3                      # 每題重複評估次數，取平均
            ).evaluate_answers()

        else:
            print(f"❌ 找不到檔案，略過評估：{output_path}")

def _countdown_sleep(seconds):
    # 倒數等待時間，並即時在同一行更新提示（適合限流保護）
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        print(f"⏳ 剩餘等待時間：{mins:02d}:{secs:02d}", end='\r', flush=True)
        time.sleep(1)
    print("⏳ 倒數結束！     ")  # 多餘空白避免前面字殘留

if __name__ == "__main__":
    # Step 1：從 DB 匯出 QA CSV
    # run_step1_export()

    # Step 2：針對每個模型產生回答
    # run_step2_generate_answers()

    # Step 3：對產生的回答進行模型自動評估
    run_step3_evaluate_answers()
