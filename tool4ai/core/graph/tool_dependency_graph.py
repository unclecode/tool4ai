import uuid
from typing import Dict, List, Any, Callable, Set, Optional
from collections import deque
import asyncio
from ..models import SubQuery, SubQueryResponse, ExecutionResult, ExecutionStatus
from ...storages import BaseStorage, JSONStorage
from ..toolkit import Toolkit
from .execution_strategy import DefaultExecutionStrategy
from .visualization import GraphVisualizer
from .result_generator import ResultGenerator


class ToolDependencyGraph:
    def __init__(
        self,
        storage: Optional[BaseStorage] = None,
        execution_strategy: Optional[DefaultExecutionStrategy] = None,
        visualizer: Optional[GraphVisualizer] = None,
        result_generator: Optional[ResultGenerator] = None,
        router: Any = None,
    ):
        self.run_id = str(uuid.uuid4())
        self.router = router
        self.storage = storage or JSONStorage()
        self.execution_strategy = execution_strategy or DefaultExecutionStrategy()
        self.visualizer = visualizer or GraphVisualizer()
        self.result_generator = result_generator or ResultGenerator()
        self.sub_queries: Dict[int, SubQuery] = {}
        self.dependency_map: Dict[int, Set[int]] = {}
        self.reverse_dependency_map: Dict[int, Set[int]] = {}
        self.results: Dict[int, Any] = {}
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.level_status: Dict[int, str] = {}
        self.graph_status: str = "pending"

    def build_dependency_structure(self, sub_query_response: SubQueryResponse):
        for sub_query in sub_query_response.sub_queries:
            self.sub_queries[sub_query.index] = sub_query
            sub_query.actionable = sub_query.tool is not None
            if sub_query.dependent_on >= 0 and sub_query.tool:
                if sub_query.index not in self.dependency_map:
                    self.dependency_map[sub_query.index] = set()
                self.dependency_map[sub_query.index].add(sub_query.dependent_on)

                if sub_query.dependent_on not in self.reverse_dependency_map:
                    self.reverse_dependency_map[sub_query.dependent_on] = set()
                self.reverse_dependency_map[sub_query.dependent_on].add(sub_query.index)

    @property
    def non_actionable_sub_queries(self) -> List[int]:
        return [index for index, sq in self.sub_queries.items() if not sq.tool]

    def get_sub_queries_by_status(self, status: ExecutionStatus) -> List[int]:
        return [index for index, sq in self.sub_queries.items() if sq.status == status]

    def get_execution_order(self, add_non_actionable=False) -> List[List[int]]:
        execution_order: List[List[int]] = []
        visited: Set[int] = set()
        queue = deque()

        root_nodes = set(
            index
            for index, sq in self.sub_queries.items()
            if sq.tool and index not in self.dependency_map
        )
        non_tool_nodes = set(
            index for index, sq in self.sub_queries.items() if not sq.actionable
        )
        queue.extend(root_nodes)

        while queue:
            level_size = len(queue)
            current_level = []

            for _ in range(level_size):
                node = queue.popleft()
                if node not in visited:
                    visited.add(node)
                    current_level.append(node)

                    if node in self.reverse_dependency_map:
                        for child in self.reverse_dependency_map[node]:
                            if child not in visited and all(
                                dep in visited
                                for dep in self.dependency_map.get(child, set())
                            ):
                                queue.append(child)

            if current_level:
                execution_order.append(current_level)

        if non_tool_nodes and add_non_actionable:
            execution_order.append(list(non_tool_nodes))
        return execution_order

    def update_graph_status(self):
        if all(status == "success" for status in self.level_status.values()):
            self.graph_status = "success"
        elif any(status == "failed" for status in self.level_status.values()):
            self.graph_status = "failed"
        elif any(status == "human" for status in self.level_status.values()):
            self.graph_status = "human"
        else:
            self.graph_status = "pending"

    async def execute(
        self,
        toolkit: Toolkit,
        context: Dict[str, Any],
        tool_maker: Any,
        final_prompt: Optional[str] = None,
        verbose: bool = False,
        generate_interim_messages: bool = False,
        add_human_failed_memory: bool = False,
        resume_from_level: int = 0,
        **kwargs,
    ) -> ExecutionResult:
        return await self.execution_strategy.execute(
            self,
            tool_functions = toolkit.tool_function_map,
            tools_info = toolkit.to_json_schema(),
            context = context,
            tool_maker = tool_maker,
            final_prompt = final_prompt,
            verbose = verbose,
            generate_interim_messages = generate_interim_messages,
            add_human_failed_memory = add_human_failed_memory,
            resume_from_level = resume_from_level,
            **kwargs,
        )

    async def _execute(
        self,
        tool_functions: Dict[str, Callable],
        tools_info: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        tool_maker: Any,
        final_prompt: Optional[str] = None,
        verbose: bool = False,
        generate_interim_messages: bool = False,
        add_human_failed_memory: bool = False,
        resume_from_level: int = 0,
    ) -> ExecutionResult:
        return await self.execution_strategy.execute(
            self,
            tool_functions,
            tools_info,
            context,
            tool_maker,
            final_prompt,
            verbose,
            generate_interim_messages,
            add_human_failed_memory,
            resume_from_level,
        )

    async def resume_execution(
        self,
        user_input: str,
        toolkit: Toolkit,
        context: Dict[str, Any],
        tool_maker: Any,
        verbose: bool = False,
        generate_interim_messages: bool = False,
        add_human_failed_memory = False,
        generate_sub_queries = False,
        classify_for_new_discussion = True,
        last_result: Optional[ExecutionResult] = None,
    ) -> ExecutionResult:
        return await self.execution_strategy.resume_execution(
            self,
            user_input = user_input,
            tool_functions = toolkit.tool_function_map,
            tools_info = toolkit.to_json_schema(),
            context = context,
            tool_maker = tool_maker,
            verbose = verbose,
            generate_interim_messages = generate_interim_messages,
            add_human_failed_memory = add_human_failed_memory,
            generate_sub_queries = generate_sub_queries,
            classify_for_new_discussion = classify_for_new_discussion,
            last_result = last_result,
        )
    
    async def _resume_execution(
        self,
        user_input: str,
        tool_functions: Dict[str, Callable],
        tools_info: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        tool_maker: Any,
        verbose: bool = False,
        generate_interim_messages: bool = False,
        add_human_failed_memory = False,
        generate_sub_queries = False,
        classify_for_new_discussion = True,
    ) -> ExecutionResult:
        return await self.execution_strategy.resume_execution(
            self,
            user_input,
            tool_functions,
            tools_info,
            context,
            tool_maker,
            verbose,
            generate_interim_messages,
            add_human_failed_memory,
            generate_sub_queries,
            classify_for_new_discussion,
        )

    def visualize(self, output_file: str = "tool_dependency_graph"):
        self.visualizer.visualize(self, output_file)

    def to_cytoscape_json(self):
        return self.visualizer.to_cytoscape_json(self)

    def generate_interactive_html(self, output_file: str = "interactive_graph.html"):
        self.visualizer.generate_interactive_html(self, output_file)

    async def save(self) -> None:
        # conver self.dependency_map and self.reverse_dependency_map to list
        list_dependency_map = {
            key: list(value) for key, value in self.dependency_map.items()
        }
        list_reverse_dependency_map = {
            key: list(value) for key, value in self.reverse_dependency_map.items()
        }
        data = {
            "run_id": self.run_id,
            "sub_queries": {
                idx: sq.model_dump() for idx, sq in self.sub_queries.items()
            },
            "dependency_map": list_dependency_map,
            "reverse_dependency_map": list_reverse_dependency_map,
            "results": self.results,
            "token_usage": self.token_usage,
            "level_status": self.level_status,
            "graph_status": self.graph_status,
        }
        await self.storage.save(self.run_id, data)

    def save_sync(self) -> None:
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async save function until it completes
            loop.run_until_complete(self.save())
        finally:
            # Close the event loop
            loop.close()

    @classmethod
    async def load(
        cls, run_id: str, storage: Optional[BaseStorage] = None
    ) -> "ToolDependencyGraph":
        storage = storage or JSONStorage()
        data = await storage.load(run_id)

        graph = cls(storage=storage)
        graph.run_id = data["run_id"]
        graph.sub_queries = {
            int(idx): SubQuery(**sq_data)
            for idx, sq_data in data["sub_queries"].items()
        }

        # Convert data["dependency_map"] and data["reverse_dependency_map"] from list to set()
        dependency_map = {
            int(key): set(value) for key, value in data["dependency_map"].items()
        }
        reverse_dependency_map = {
            int(key): set(value)
            for key, value in data["reverse_dependency_map"].items()
        }
        graph.dependency_map = dependency_map
        graph.reverse_dependency_map = reverse_dependency_map
        graph.results = data["results"]
        graph.token_usage = data["token_usage"]
        graph.level_status = data["level_status"]
        graph.graph_status = data["graph_status"]

        return graph

    @classmethod
    def load_sync(
        cls, run_id: str, storage: Optional[BaseStorage] = None
    ) -> "ToolDependencyGraph":
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async load function until it completes
            graph = loop.run_until_complete(cls.load(run_id, storage))
            return graph
        finally:
            # Close the event loop
            loop.close()

    async def delete(self) -> None:
        await self.storage.delete(self.run_id)

    def reset(self) -> None:
        self.sub_queries = {}
        self.dependency_map = {}
        self.reverse_dependency_map = {}
        self.results = {}
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.level_status = {}
        self.graph_status = "pending"

    def _update_token_usage(self, usage):
        for key in self.token_usage:
            self.token_usage[key] += usage.get(key, 0)

    def update_token_usage(self, usage):
        for key in self.token_usage:
            self.token_usage[key] += usage.get(key, 0)

    def get_token_usage(self):
        return self.token_usage

    def get_results(self):
        return self.results
