# 📊 模型回答準確率 - 測試模組（Evaluation Module）

本模組為主專案「RAG 文件檢索聊天機器人」的延伸測試工具，旨在自動化產生問答資料集、執行不同語言模型的回答任務，並量化其準確率與相似度表現。

---

## 🎯 功能目標

- 從文件自動產出問答資料集
- 呼叫不同 LLM 模型進行回答（支援內部或外部 API）
- 評估模型回答與標準答案的相似度
- 匯出結果 CSV，供後續分析與報表使用

---
## 📊 測試結果（2025/04/22）  
> 評估模型：`Gemma3_27b`
- `Gemma3_27b` 在一致性與穩定性表現最佳，平均得分最高。
- `Deepseek_14b_QwenDistill` 輕量、效率最高。
- `QWQ_32b` 推理模型，能回答複雜問題(因果推理)，但有時會想太多、效率慢。

| No.  | LLM                          | Test1 | Test2 | Test3 | Avg.  |
|------|------------------------------|-------|-------|-------|-------|
| ⭐ 1 | Gemma3_27b                   | 38    | 37    | 38    | 37.7  |
| ⭐ 2  | Deepseek_14b_QwenDistill     | 39    | 37    | 36    | 37.3  |
| ⭐ 3  | Gemma2_27b                   | 36    | 38    | 38    | 37.3  |
| 4    | QWQ_32b                      | 39    | 36    | 36    | 37.0  |
| 5    | Deepseek_7b                  | 38    | 33    | 35    | 35.3  |
| 6    | Phi4_14b                     | 33    | 35    | 36    | 34.7  |
| 7    | Gemma3_4b                    | 33    | 33    | 34    | 33.3  |
| 8    | Taiwan_LLaMA3_8b_Instruct    | 33    | 33    | 31    | 32.3  |
| 9    | Mistral_7b_Instruct          | 33    | 29    | 33    | 31.7  |
| 10   | LLaMA3_2_Latest              | 28    | 33    | 31    | 30.7  |


## ⚙️ 執行步驟

### 🔁 一鍵全流程
```bash
python evaluation/run_llm_evaluation.py
```

### 🧪 分步執行（可搭配 import 呼叫）
```python
from evaluation import run_llm_evaluation

# Step 1：從 DB 匯出 QA CSV（如未註解）
run_llm_evaluation.run_step1_export()

# Step 2：對多個模型產生回答
run_llm_evaluation.run_step2_generate_answers()

# Step 3：執行模型回答評估與比對
run_llm_evaluation.run_step3_evaluate_answers()
```

---
## 📂 專案結構

```
📂 evaluation/                         # 模型回答評估模組（本模組主要工作目錄）  
├── 📦 run_llm_evaluation.py           # 📌 主流程入口，執行「匯出、生成、評估」三階段  
├── 📦 export_data.py                  # 匯出 DB 中 chunks 與結果  
├── 📦 gen_qa.py                       # 從 chunks 自動產生問答配對資料  
├── 📦 eval_answer.py                  # 使用評審模型評估回答正確性  
  
📂 apis/  
├── 📦 llm_api.py                      # 管理所有可用 LLM 模型（內部 / 外部 API）  
  
📂 mockdata/                           # 🔧 測試輸出資料夾（含每模型回答與評估結果）  
├── 📜 QAData_retrievedContent.csv     # 從 DB 匯出的原始 QA 資料  
├── 📜 QAData_goldenChunk.csv          # 黃金標準答案 QA 資料  
├── 📂 {模型名稱}/  
│   ├── 📜 test{n}_{模型}_results.csv              # 模型產生的回答結果  
│   └── 📜 test{n}_{模型}_evaluate_{eval_模型}.csv # 模型回答與黃金答案的評估對照表  
  
📂 data/  
└── 📜 user/{user_name}/{user_name}.db # 使用者對應的 QA 資料來源 DB  
```

---

## 🧠 技術說明

- 使用 `BGE-M3`, `Cosine Similarity` 搜尋相關文件 (Chunks)
- 可設定不同評審模型（如 GPT-4o、Gemma 等）， 多數決(3次)判定回答是否正確
- 具備限流保護（每個模型評估間可暫停指定秒數） 
- 流程圖  
![eval_workflow.png](eval_workflow.png) 
---

## 📌 注意事項

- 須先設定 `.env` 檔案與 `apis/llm_api.py`（`api_base`、`api_key`）以指定模型來源
- 輸入資料建議使用結構化 PDF 或已分段文本，能提高問答產出準確度

---

## 👩‍💻 模組貢獻者

- 本模組由 [Valerie Chen](mailto:valerie7331@gmail.com) 製作與維護
