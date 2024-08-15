# tool4ai/storages/base_storage.py

import abc
from typing import Dict, Any

class BaseStorage(abc.ABC):
    @abc.abstractmethod
    async def save(self, run_id: str, data: Dict[str, Any]) -> None:
        pass

    @abc.abstractmethod
    async def load(self, run_id: str) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    async def delete(self, run_id: str) -> None:
        pass