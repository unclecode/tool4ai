# tool4ai/storages/json_storage.py

import os
import json
import asyncio
from typing import Dict, Any
from .base_storage import BaseStorage

class JSONStorage(BaseStorage):
    def __init__(self):
        self.storage_path = os.path.expanduser("~/.tool4ai/storage/json")
        os.makedirs(self.storage_path, exist_ok=True)

    def _get_file_path(self, run_id: str) -> str:
        return os.path.join(self.storage_path, f"{run_id}.json")

    async def save(self, run_id: str, data: Dict[str, Any]) -> None:
        file_path = self._get_file_path(run_id)
        
        def _save():
            with open(file_path, 'w') as f:
                json.dump(data, f)

        await asyncio.to_thread(_save)

    async def load(self, run_id: str) -> Dict[str, Any]:
        file_path = self._get_file_path(run_id)
        
        def _load():
            if not os.path.exists(file_path):
                raise KeyError(f"No data found for run_id: {run_id}")
            with open(file_path, 'r') as f:
                return json.load(f)

        return await asyncio.to_thread(_load)

    async def delete(self, run_id: str) -> None:
        file_path = self._get_file_path(run_id)
        
        def _delete():
            if os.path.exists(file_path):
                os.remove(file_path)

        await asyncio.to_thread(_delete)

async def main():
    storage = JSONStorage()
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