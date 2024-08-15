# tool4ai/storages/lmdb_storage.py

import os
import json
import lmdb
import asyncio
from typing import Dict, Any
from .base_storage import BaseStorage

class LMDBStorage(BaseStorage):
    def __init__(self):
        self.db_path = os.path.expanduser("~/.tool4ai/storage/lmdb")
        os.makedirs(self.db_path, exist_ok=True)
        self.env = lmdb.open(self.db_path, map_size=1024*1024*1024)  # 1GB max

    async def save(self, run_id: str, data: Dict[str, Any]) -> None:
        def _save():
            with self.env.begin(write=True) as txn:
                txn.put(run_id.encode(), json.dumps(data).encode())

        await asyncio.to_thread(_save)

    async def load(self, run_id: str) -> Dict[str, Any]:
        def _load():
            with self.env.begin() as txn:
                data = txn.get(run_id.encode())
                if data is None:
                    raise KeyError(f"No data found for run_id: {run_id}")
                return json.loads(data.decode())

        return await asyncio.to_thread(_load)

    async def delete(self, run_id: str) -> None:
        def _delete():
            with self.env.begin(write=True) as txn:
                txn.delete(run_id.encode())

        await asyncio.to_thread(_delete)

async def main():
    storage = LMDBStorage()
    run_id = "test_run_1"
    test_data = {"key": "value", "nested": {"a": 1, "b": 2}}

    await storage.save(run_id, test_data)
    print(f"Saved data for run_id: {run_id}")

    loaded_data = await storage.load(run_id)
    print(f"Loaded data for run_id: {run_id}")
    print(loaded_data)

    await storage.delete(run_id)
    print(f"Deleted data for run_id: {run_id}")

    try:
        await storage.load(run_id)
    except KeyError as e:
        print(f"Expected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())