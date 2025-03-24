from pathlib import Path

class FilePaths:
    def __init__(self, base_dir=None):
        """
        初始化時設置基本路徑，如果未提供 base_dir 則默認為 'data' 資料夾。
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = Path(__file__).resolve().parent.parent / 'data'

    def get_tmp_dir(self, username, conversation_id):
        """
        獲取暫存目錄 (tmp_dir) 的路徑。
        """
        return self.base_dir / 'user' / username / conversation_id / 'tmp'

    def get_local_vector_store_dir(self, username, conversation_id):
        """
        獲取向量存儲目錄 (local_vector_store_dir) 的路徑。
        """
        return self.base_dir / 'user' / username / conversation_id / 'vector_store'

    def get_output_dir(self):
        """
        獲取輸出目錄 (output_dir) 的路徑。
        """
        return self.base_dir / 'output'

    def get_user_records_dir(self, username):
        """
        獲取使用者紀錄目錄 (user_records_dir) 的路徑。
        """
        return self.base_dir / 'user' / username

    def get_developer_dir(self):
        """
        獲取開發者目錄 (dev_ops_dir) 的路徑。
        """
        return self.base_dir / 'developer'
