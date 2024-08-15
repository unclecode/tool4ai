# execution_strategy.py
from typing import Dict, List, Any, Callable, Optional
from ..models import ExecutionStatus, ExecutionResult, SubQuery, SubQueryResponse
from ...toolmakers import ToolMaker
import asyncio
import json
from abc import ABC, abstractmethod
import copy

class ExecutionStrategy(ABC):
    def __init__(self):
        self.last_context = None
        pass
    @abstractmethod
    async def execute(
        self,
        graph,
        tool_functions: Dict[str, Callable],
        tools_info: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        tool_maker: ToolMaker,
        final_prompt: Optional[str] = None,
        verbose: bool = False,
        generate_interim_messages: bool = False,
        add_human_failed_memory: bool = False,
        resume_from_level: int = 0,
    ) -> ExecutionResult:
        pass

    @abstractmethod
    async def resume_execution(
        self,
        graph,
        user_input: str,
        tool_functions: Dict[str, Callable],
        tools_info: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        tool_maker: ToolMaker,
        verbose: bool = False,
        generate_interim_messages: bool = False,
        add_human_failed_memory:bool = False,
    ) -> ExecutionResult:
        pass

    async def _execute_level(
        self,
        graph,
        level: int,
        indices: List[int],
        tool_functions: Dict[str, Callable],
        tools_info: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        tool_maker: ToolMaker,
    ) -> List[Dict[str, Any]]:
        pass

    async def _execute_sub_query(
        self,
        graph,
        index: int,
        tool_functions: Dict[str, Callable],
        tools_info: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        tool_maker: ToolMaker,
    ) -> Dict[str, Any]:
        pass

class DefaultExecutionStrategy(ExecutionStrategy):
    
    async def execute(
        self,
        graph,
        tool_functions: Dict[str, Callable],
        tools_info: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        tool_maker: ToolMaker,
        final_prompt: Optional[str] = None,
        verbose: bool = False,
        generate_interim_messages: bool = False,
        add_human_failed_memory: bool = False,
        resume_from_level: int = 0,
    ) -> ExecutionResult:
        execution_order = graph.get_execution_order()
        memory = context.get("memory", [])

        try:
            for level, indices in enumerate(
                execution_order[resume_from_level:], start=resume_from_level
            ):
                if verbose:
                    print(f"Executing level: {level}")

                if graph.level_status.get(level) == "success":
                    continue

                indices_to_execute = [
                    idx for idx in indices if graph.sub_queries[idx].status != "success"
                ]
                level_results = await self._execute_level(
                    graph,
                    level,
                    indices_to_execute,
                    tool_functions,
                    tools_info,
                    context,
                    tool_maker,
                )
                level_sub_queries = [item["sub_query"] for item in level_results]

                graph.level_status[level] = "success"

                # update level status
                for sq in level_sub_queries:
                    if sq.status in ["failed", "human"]:
                        graph.level_status[level] = sq.status
                        break

                # Update memory
                for item in level_results:
                    sq, memory_entry = item["sub_query"], item["memory"]
                    if sq.status == "success":
                        memory.extend(memory_entry)
                if add_human_failed_memory:
                    for item in level_results:
                        sq, memory_entry = item["sub_query"], item["memory"]
                        if sq.status in ["failed", "human"]:
                            memory.extend(memory_entry)

                # update graph status
                graph.update_graph_status()

                if graph.level_status[level] in ["failed", "human"]:
                    if generate_interim_messages:
                        interim_message, usages = (
                            await graph.result_generator.generate_interim_message(
                                tool_maker, level_sub_queries, context
                            )
                        )
                        context['memory'].append({"role": "assistant", "content": interim_message})
                        context["interim_message"] = interim_message
                        graph.update_token_usage(usages)
                    break

                context["memory"] = memory

            if final_prompt and graph.graph_status == "success":
                final_response, usage = (
                    await graph.result_generator.generate_final_response(
                        context, tool_maker, final_prompt
                    )
                )
                memory.append({"role": "assistant", "content": final_response})
                graph.update_token_usage(usage)

            self.last_context = context
            
            # check for all orphan sub queries, udated their index a
            for index, sub_query in graph.sub_queries.items():
                if sub_query.is_orphan:
                    sub_query.is_orphan = False
                    sub_query.index = len(graph.sub_queries)
                    graph.sub_queries[len(graph.sub_queries)] = sub_query
            
            return ExecutionResult(
                status=ExecutionStatus(graph.graph_status),
                message=f"Execution {graph.graph_status}.",
                memory=memory,
                sub_queries=list(graph.sub_queries.values()),
            )

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                message=f"Execution failed: {str(e)}",
                memory=memory,
                sub_queries=list(graph.sub_queries.values()),
                error_info={"error_type": type(e).__name__, "error_message": str(e)},
            )
    
    async def _execute_level(
        self,
        graph,
        level: int,
        indices: List[int],
        tool_functions: Dict[str, Callable],
        tools_info: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        tool_maker: ToolMaker,
    ) -> List[Dict[str, Any]]:
        level_tasks = [
            self._execute_sub_query(
                graph, index, tool_functions, tools_info, context, tool_maker
            )
            for index in indices
        ]
        level_results = await asyncio.gather(*level_tasks)
        # Flatten the list of lists into a single list
        return [item for sublist in level_results for item in sublist]

    async def resume_execution(
        self,
        graph,
        user_input: str,
        tool_functions: Dict[str, Callable],
        tools_info: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        tool_maker: ToolMaker,
        verbose: bool = False,
        generate_interim_messages: bool = False,
        add_human_failed_memory = False,
        generate_sub_queries = False,
        classofy_for_new_discussion = True,
    ) -> ExecutionResult:
        if classofy_for_new_discussion:
            classification, usage = await graph.result_generator.classify_user_input(
                tool_maker, user_input, context
            )
            graph.update_token_usage(usage)

            if classification == "new_discussion":
                return ExecutionResult(
                    status=ExecutionStatus.NEW_DISCUSSION,
                    message="User started a new discussion. Execution not resumed.",
                    memory=context.get("memory", []),
                    sub_queries=list(graph.sub_queries.values()),
                )

        resume_from_level = max(graph.level_status.keys()) if graph.level_status else 0
        
        execution_order = graph.get_execution_order()
        
        level_sub_queries =[graph.sub_queries[idx] for idx in execution_order[resume_from_level]]

        if generate_sub_queries:
            sub_queries : SubQueryResponse = graph.router.gen_subquery(graph, user_input)
            for sub_query in level_sub_queries:
                if sub_query.status in ["failed", "human"]:
                    sub_query.status = "success"
                    sub_query.issue = ""
            # Append new sub queries to the same level
            graph.sub_queries[resume_from_level].extend(sub_queries.sub_queries)
        else:
            for sub_query in level_sub_queries:
                if sub_query.status in ["failed", "human"]:
                    sub_query.status = "pending"
                    sub_query.issue = None
                    sub_query.result = None
                    sub_query.task = user_input
                    sub_query.sub_query = user_input

        return await self.execute(
            graph,
            tool_functions,
            tools_info,
            context or self.last_context,
            tool_maker,
            verbose=verbose,
            generate_interim_messages=generate_interim_messages,
            add_human_failed_memory = add_human_failed_memory,
            resume_from_level=resume_from_level,
        )

    async def _execute_sub_query(
        self,
        graph,
        index: int,
        tool_functions: Dict[str, Callable],
        tools_info: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        tool_maker: ToolMaker,
    ) -> List[Dict[str, Any]]:
        original_sub_query = graph.sub_queries[index]
        results = []

        try:
            filtered_tools_info = {
                tool: info
                for tool, info in tools_info.items()
                if info["name"] == original_sub_query.tool
            }

            message, usage = await tool_maker.make_tools(
                original_sub_query.task, filtered_tools_info, context
            )
            graph.update_token_usage(usage)
            
            if len(message["tool_calls"]) > 1:
                print(f"Multiple tool calls in a single message: {message}")
                

            is_orphan = False
            for tool_call in message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_id = tool_call["id"]
                arguments = json.loads(tool_call["function"]["arguments"])

                # Create a new sub_query for each tool call
                sub_query = copy.deepcopy(original_sub_query) if not is_orphan else original_sub_query
                sub_query.tool = tool_name
                sub_query.is_orphan = is_orphan
                is_orphan = True

                if tool_name in tool_functions:
                    result = await tool_functions[tool_name](arguments)

                    result_dict = json.loads(result)
                    sub_query.status = result_dict["status"]

                    memory_entry = [
                        {"role": "user", "content": sub_query.task},
                        message,
                        {"role": "tool", "tool_call_id": tool_id, "name": tool_name},
                    ]

                    if sub_query.status == "success":
                        sub_query.result = json.dumps(result_dict.get("return", {}))
                        sub_query.issue = None
                        memory_entry[-1]["content"] = sub_query.result
                    else:  # 'failed' or 'human'
                        sub_query.result = result_dict.get("message", "")
                        sub_query.issue = sub_query.result
                        memory_entry[-1]["content"] = sub_query.result
                        memory_entry[-1]["status"] = sub_query.status

                    results.append({
                        "index": index,
                        "sub_query": sub_query,
                        "status": sub_query.status,
                        "memory": memory_entry,
                    })
                else:
                    raise ValueError(f"No function found for tool {tool_name}")

        except Exception as e:
            print(f"Error executing {original_sub_query.task}: {str(e)}")
            original_sub_query.status = "failed"
            original_sub_query.result = json.dumps({"error": str(e)})
            original_sub_query.issue = str(e)
            results.append({
                "index": index,
                "sub_query": original_sub_query,
                "status": "failed",
                "memory": [
                    {"role": "error", "content": f"Error in {original_sub_query.tool}: {str(e)}"}
                ],
            })

        return results