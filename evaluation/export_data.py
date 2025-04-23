# rag_data_exporter.py
import os
import sqlite3
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

class RAGDataExporter:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def export_to_csv(self, output_path: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 🔹 多撈一欄 chunk_id
        cursor.execute("""
            SELECT rh.id, rh.query, rd.doc_index, rd.chunk_id, rd.content
            FROM rag_history rh
            JOIN retrieved_docs rd ON rd.rag_history_id = rh.id
            WHERE rd.doc_index BETWEEN 1 AND 5
            ORDER BY rh.id, rd.doc_index
        """)

        rows = cursor.fetchall()
        conn.close()

        # 🔹 每個 query 對應多筆內容與 chunk_id
        query_docs = defaultdict(lambda: {"id": None, "contents": [], "chunk_ids": []})
        for rag_id, query, doc_index, chunk_id, content in rows:
            query_docs[query]["id"] = rag_id
            query_docs[query]["contents"].append(content.strip())
            query_docs[query]["chunk_ids"].append(str(chunk_id))  # ➕ 儲存 chunk_id（轉為字串）

        # 🔹 組裝輸出紀錄，加入 "Chunk_ID" 欄位（以逗號分隔）
        records = [
            {
                "ID": data["id"],
                "Question": query,
                "DocContents": "\n\n".join(data["contents"]),
                "Chunk_ID": ", ".join(data["chunk_ids"])  # ✅ 使用逗號分隔
            }
            for query, data in query_docs.items()
        ]

        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\u2705  step1 匯出完成：{output_path}")
