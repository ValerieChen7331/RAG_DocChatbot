#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calc_recall.py  ‧ RAG Retriever 評估工具（優化版）
==================================================

【工作流程說明】
此工具用於評估 RAG (Retrieval-Augmented Generation) 系統檢索效能，流程如下：
1. 讀取黃金標準資料集 (CSV)，獲取正確的檢索文檔 ID
2. 讀取系統實際檢索結果 (SQLite 資料庫)
3. 比對實際檢索與黃金標準的重疊情況，計算召回率
4. 產生詳細評估報告 (CSV)，包含每筆查詢的召回率和整體統計指標

【輸出表格欄位說明】
* query_id：查詢ID，來自黃金標準CSV
* rag_history_id：RAG歷史記錄ID，與資料庫對應
* question：原始問題文本
* golden_chunk_text：黃金標準區塊文本內容
* golden_chunk_ids：黃金標準區塊ID列表 (逗號分隔)
* llm_option：使用的大型語言模型選項
* rewritten_query：系統重寫後的查詢
* ai_response：AI生成的回應
* timestamp：處理時間戳
* retrieved_chunk_ids：系統實際檢索的區塊ID列表 (逗號分隔)
* retrieved_chunk_texts：系統實際檢索的區塊文本 (逗號分隔)
* missed_chunk_ids：未被檢索到的黃金標準區塊 (逗號分隔)
* golden_chunk_total：黃金標準區塊總數
* hit_count：命中的區塊數量
* recall：召回率 (命中數/黃金標準總數)

【評估指標說明】
* Micro-average Recall：總體命中數除以總體黃金標準數 (∑hits/∑golds)
* Macro-average Recall：各查詢召回率的平均值 (avg(recall))

【重點更新】
1. **query_id** 一律取自 CSV 的 `ID` 欄位（需保證為 `rag_history.id`）。
2. 另外新增 `rag_history_id` 欄位，與 `retrieved_docs.rag_history_id` 完全相同，方便檢查對齊。
   ⇒ 若 CSV 的 `ID` 非純數字，請先與資料庫對映。
3. 所有欄位維持 snake_case。
4. 終端機照常輸出 Micro / Macro Recall。
5. 增強錯誤處理與日誌記錄，提高程式穩定性。
6. 統一使用逗號(,)作為所有多值欄位的分隔符號。
"""

from __future__ import annotations  # 允許在類型註解中引用自身類型

import argparse  # 命令行參數解析
import sqlite3  # SQLite數據庫連接
import logging  # 日誌記錄
import time  # 時間計算
from collections import defaultdict  # 預設字典，處理鍵不存在的情況
from pathlib import Path  # 使用Path物件處理檔案路徑，比字串更安全
from typing import Dict, Set, List, Tuple, Optional  # 類型提示
import pandas as pd  # 用於數據處理和CSV操作

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---------- 常數定義（依實際資料結構調整） ---------------------------
# CSV檔案的欄位名稱
CSV_COL_QUERY_ID = "ID"  # ← 必須對應 rag_history.id
CSV_COL_QUESTION = "Question"  # 問題文本欄位
CSV_COL_GOLDEN_IDS = "Golden_ID"  # 黃金標準區塊ID欄位 (輸入格式：逗號分隔)
CSV_COL_GOLDEN_TEXT = "DocContents"  # 文檔內容欄位

# 資料分隔符號
CHUNK_SEPARATOR = ","  # 區塊ID與文本的統一分隔符號（逗號）

# 資料庫表格名稱
TABLE_RAG = "rag_history"  # RAG歷史記錄表
TABLE_RETR = "retrieved_docs"  # 檢索文檔表
TABLE_CHAT = "chat_history"  # 聊天歷史表

# 檔案路徑設定
BASE_DIR = Path(r"D:\10_RAG\08_RAG_DocChatbot_evaluate_Reranker")  # 基礎目錄
DEFAULT_CSV = BASE_DIR / "mockdata" / "QAData_goldenChunk.csv"  # 預設黃金標準CSV檔案
DEFAULT_DB = BASE_DIR / "data" / "user" / "Guest" / "Guest.db"  # 預設數據庫文件
DEFAULT_OUT = BASE_DIR / "mockdata" / "retrieval_eval_9.csv"  # 預設輸出CSV檔案

# 自定義類型別名，提高代碼可讀性
GoldenMap = Dict[str, Set[str]]  # 查詢ID到黃金標準區塊ID集合的映射
TextMap = Dict[str, str]  # 查詢ID到文本的映射
RagDict = Dict[int, Dict[str, str]]  # RAG歷史ID到記錄信息的映射
RetMap = Dict[int, List[Tuple[str, str]]]  # RAG歷史ID到檢索結果的映射


# ---------------------------------------------------------

def fetch_df(db: Path, sql: str) -> pd.DataFrame:
    """
    從SQLite數據庫讀取資料並返回DataFrame

    參數:
        db: 數據庫文件路徑
        sql: SQL查詢語句

    返回:
        pandas DataFrame包含查詢結果
    """
    try:
        with sqlite3.connect(db) as con:  # 建立數據庫連接並確保自動關閉
            logger.debug(f"執行SQL查詢: {sql}")
            return pd.read_sql_query(sql, con)  # 執行SQL查詢並將結果轉為DataFrame
    except sqlite3.Error as e:
        logger.error(f"資料庫查詢錯誤: {e}")
        logger.error(f"問題SQL: {sql}")
        raise


def load_golden(path: Path) -> Tuple[GoldenMap, TextMap, TextMap]:
    """
    載入黃金標準CSV檔案，提取查詢ID、黃金標準區塊ID和問題文本

    參數:
        path: CSV檔案路徑

    返回:
        g_map: 字典 {查詢ID: 黃金標準區塊ID集合}
        g_text: 字典 {查詢ID: 黃金標準文本}
        q_map: 字典 {查詢ID: 問題文本}
    """
    logger.info(f"載入黃金標準資料: {path}")

    try:
        df = pd.read_csv(path)  # 讀取CSV檔案
        logger.info(f"成功載入CSV檔案，共 {len(df)} 行")
    except Exception as e:
        logger.error(f"CSV檔案載入失敗: {e}")
        raise

    g_map: GoldenMap = {}  # 黃金標準ID映射
    g_text: TextMap = {}  # 黃金標準文本映射
    q_map: TextMap = {}  # 問題文本映射

    # 驗證必要欄位是否存在
    required_columns = [CSV_COL_QUERY_ID, CSV_COL_GOLDEN_IDS]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"CSV檔案缺少必要欄位: {', '.join(missing_columns)}")
        raise ValueError(f"CSV檔案格式錯誤，缺少欄位: {', '.join(missing_columns)}")

    for _, r in df.iterrows():  # 遍歷DataFrame的每一行
        qid = str(r[CSV_COL_QUERY_ID]).strip()  # 獲取並清理查詢ID (必須等同 rag_history.id)

        if not qid:  # 跳過空ID
            logger.warning(f"跳過空查詢ID的行")
            continue

        # 從Golden_ID欄位分割多個ID，轉為集合，移除空白項
        # 注意：輸入可能使用逗號分隔，但我們統一輸出為三豎線分隔
        g_ids = {cid.strip() for cid in str(r[CSV_COL_GOLDEN_IDS]).split(',') if cid.strip()}

        g_map[qid] = g_ids  # 存儲查詢ID對應的黃金標準區塊ID集合
        g_text[qid] = str(r.get(CSV_COL_GOLDEN_TEXT, ''))  # 存儲黃金標準文本，如不存在則空字符串
        q_map[qid] = str(r.get(CSV_COL_QUESTION, ''))  # 存儲問題文本，如不存在則空字符串

    logger.info(f"共載入 {len(g_map)} 個有效查詢")
    return g_map, g_text, q_map  # 返回三個映射字典


def load_db_parts(db: Path) -> Tuple[RagDict, RetMap, Dict[str, str]]:
    """
    從數據庫加載RAG歷史、檢索文檔和聊天歷史相關資料

    參數:
        db: 數據庫文件路徑

    返回:
        rag_dict: RAG歷史記錄 {id: 記錄信息字典}
        ret_map: 檢索文檔映射 {rag_history_id: [(chunk_id, content), ...]}
        conv2opt: 對話ID與LLM選項的映射 {conversation_id: llm_option}
    """
    logger.info(f"連接資料庫: {db}")

    if not db.exists():
        logger.error(f"資料庫檔案不存在: {db}")
        raise FileNotFoundError(f"找不到資料庫檔案: {db}")

    # 查詢RAG歷史記錄表，獲取重要欄位
    logger.info(f"讀取RAG歷史記錄")
    df_rag = fetch_df(db, f"SELECT id,conversation_id,query,rewritten_query,response,timestamp FROM {TABLE_RAG}")
    logger.info(f"載入 {len(df_rag)} 筆RAG歷史記錄")
    rag_dict = df_rag.set_index('id').to_dict('index')  # 轉換為以id為鍵的字典

    # 查詢檢索文檔表，按doc_index排序
    logger.info(f"讀取檢索文檔")
    df_ret = fetch_df(db, f"SELECT rag_history_id,chunk_id,content FROM {TABLE_RETR} ORDER BY doc_index")
    ret_map: RetMap = defaultdict(list)  # 使用defaultdict避免鍵不存在的錯誤

    # 將檢索文檔按rag_history_id分組
    chunk_count = 0
    for _, r in df_ret.iterrows():
        ret_map[r['rag_history_id']].append((str(r['chunk_id']), str(r['content'])))
        chunk_count += 1
    logger.info(f"載入 {chunk_count} 個檢索文檔，涉及 {len(ret_map)} 個查詢")

    # 查詢聊天歷史表，獲取LLM選項
    logger.info(f"讀取聊天歷史")
    df_chat = fetch_df(db, f"SELECT DISTINCT conversation_id,llm_option FROM {TABLE_CHAT}")
    conv2opt = df_chat.set_index('conversation_id')['llm_option'].to_dict()  # 轉換為字典
    logger.info(f"載入 {len(conv2opt)} 個對話配置")

    return rag_dict, ret_map, conv2opt  # 返回三個數據結構


def build_df(
        g_map: GoldenMap,
        g_text: TextMap,
        q_map: TextMap,
        rag_dict: RagDict,
        ret_map: RetMap,
        conv2opt: Dict[str, str]
) -> Tuple[pd.DataFrame, float, float]:
    """
    構建評估結果DataFrame，計算召回率

    參數:
        g_map: 黃金標準區塊ID映射
        g_text: 黃金標準文本映射
        q_map: 問題文本映射
        rag_dict: RAG歷史記錄
        ret_map: 檢索文檔映射
        conv2opt: 對話與LLM選項映射

    返回:
        df: 評估結果DataFrame
        micro: 微平均召回率 (總命中數/總黃金標準數)
        macro: 宏平均召回率 (各查詢召回率的平均)
    """
    logger.info("開始構建評估結果")
    start_time = time.time()

    rows = []  # 儲存結果行
    total_hit = 0  # 總命中計數
    total_gold = 0  # 總黃金標準計數
    missing_rag_ids = 0  # 統計找不到RAG記錄的查詢數

    # 遍歷每個查詢ID及其黃金標準區塊集合
    for qid, golden_ids in g_map.items():
        # 查詢ID轉為整數(如果是數字)，否則為None
        rag_id = int(qid) if qid.isdigit() else None

        # 獲取RAG記錄，不存在則為空字典
        rag = rag_dict.get(rag_id, {})
        if not rag and rag_id is not None:
            missing_rag_ids += 1
            logger.warning(f"在資料庫中找不到查詢ID {qid} 的RAG記錄")

        # 獲取檢索的文檔，不存在則為空列表
        topk = ret_map.get(rag_id, [])

        # 提取檢索文檔的ID和文本
        top_ids = [cid for cid, _ in topk]
        top_txts = [txt for _, txt in topk]

        # 計算命中數 (黃金標準與檢索結果的交集大小)
        retrieved_ids_set = set(top_ids)
        hit = len(golden_ids & retrieved_ids_set)
        missed_ids = golden_ids - retrieved_ids_set

        # 計算召回率
        recall = hit / len(golden_ids) if golden_ids else 0.0

        # 創建結果字典
        rows.append({
            'query_id': qid,  # 查詢ID
            'rag_history_id': rag_id if rag_id is not None else '',  # RAG歷史ID
            'question': q_map.get(qid, rag.get('query', '')),  # 問題文本
            'golden_chunk_text': g_text.get(qid, ''),  # 黃金標準文本
            'golden_chunk_ids': CHUNK_SEPARATOR.join(sorted(golden_ids)),  # 黃金標準區塊ID
            'llm_option': conv2opt.get(rag.get('conversation_id', ''), ''),  # LLM選項
            'rewritten_query': rag.get('rewritten_query', ''),  # 重寫後的查詢
            'ai_response': rag.get('response', ''),  # AI回應
            'timestamp': rag.get('timestamp', ''),  # 時間戳
            'retrieved_chunk_ids': CHUNK_SEPARATOR.join(top_ids),  # 檢索的區塊ID
            'retrieved_chunk_texts': CHUNK_SEPARATOR.join(top_txts),  # 檢索的文本
            'missed_chunk_ids': CHUNK_SEPARATOR.join(sorted(missed_ids)),  # 未命中區塊ID
            'golden_chunk_total': len(golden_ids),  # 黃金標準總數
            'hit_count': hit,  # 命中數
            'recall': recall  # 召回率
        })

        # 累計總命中數和總黃金標準數
        total_hit += hit
        total_gold += len(golden_ids)

    # 計算微平均召回率 (總命中數/總黃金標準數)
    micro = total_hit / total_gold if total_gold else 0.0

    # 計算宏平均召回率 (各查詢召回率的平均)
    macro = sum(r['recall'] for r in rows) / len(rows) if rows else 0.0

    elapsed = time.time() - start_time
    logger.info(f"評估完成，處理 {len(rows)} 個查詢，耗時 {elapsed:.2f} 秒")
    logger.info(f"遺漏的RAG記錄數量: {missing_rag_ids}")
    logger.info(f"Micro-average Recall: {micro:.4f}, Macro-average Recall: {macro:.4f}")

    return pd.DataFrame(rows), micro, macro  # 返回DataFrame和兩個平均召回率


def main():
    """
    主函數：解析命令行參數，執行評估流程，保存結果
    """
    # 創建參數解析器
    ap = argparse.ArgumentParser(description='RAG檢索系統評估工具')
    ap.add_argument('--csv', default=DEFAULT_CSV, help='黃金標準CSV檔案路徑')
    ap.add_argument('--db', default=DEFAULT_DB, help='數據庫檔案路徑')
    ap.add_argument('--output', default=DEFAULT_OUT, help='輸出CSV檔案路徑')
    ap.add_argument('--verbose', '-v', action='store_true', help='顯示詳細日誌')
    args = ap.parse_args()  # 解析命令行參數

    # 如果需要詳細日誌，調整日誌等級
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("啟用詳細日誌模式")

    try:
        # 載入黃金標準資料
        g_map, g_text, q_map = load_golden(Path(args.csv))

        # 載入數據庫資料
        rag_dict, ret_map, conv2opt = load_db_parts(Path(args.db))

        # 建立評估結果及計算召回率
        df, micro, macro = build_df(g_map, g_text, q_map, rag_dict, ret_map, conv2opt)

        # 準備輸出路徑
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)  # 創建輸出目錄（如不存在）

        # 保存結果為CSV
        df.to_csv(out_p, index=False, encoding='utf-8-sig')  # 使用utf-8-sig保存中文

        # 輸出結果摘要
        print(f"\n✅ 輸出 → {out_p} (rows={len(df)})")
        print(f"Micro-average Recall : {micro:.4f}")  # 微平均召回率 (4位小數)
        print(f"Macro-average Recall : {macro:.4f}")  # 宏平均召回率 (4位小數)

    except ValueError as e:
        logger.error(f"資料格式錯誤: {e}")
        print(f"\n❌ 錯誤: {e}")
        return 1
    except FileNotFoundError as e:
        logger.error(f"找不到檔案: {e}")
        print(f"\n❌ 錯誤: {e}")
        return 1
    except Exception as e:
        logger.error(f"發生未預期的錯誤: {e}", exc_info=True)
        print(f"\n❌ 未預期錯誤: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit_code = main()  # 作為主程式運行時執行main函數
    exit(exit_code)  # 返回適當的退出碼