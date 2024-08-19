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
        self.issue = None
        self.help = None
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
        **kwargs,
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
        **kwargs,
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
        **kwargs,
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
        **kwargs,
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
        **kwargs,
    ) -> ExecutionResult:
        execution_order = graph.get_execution_order()
        memory = context.get("memory", [])
        
        sub_query_need_attention = None
        pasued_level = -1

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
                    if sq.status != "success":
                        graph.level_status[level] = sq.status
                        sub_query_need_attention = sq
                        pasued_level = level
                        break

                # Update memory with successful sub-queries
                for item in level_results:
                    sq = item["sub_query"]
                    if sq.status == "success":
                        memory.extend(sq.internal_memory)
                
                # Check if any failed sub-query 
                self.issue = ["\n".join(sq.issue) for sq in level_sub_queries if sq.issue]
                self.help = ["\n".join(sq.help) for sq in level_sub_queries if sq.help]
                
                # update graph status
                graph.update_graph_status()

                if graph.level_status[level] != "success":
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
    
            
            return ExecutionResult(
                status=ExecutionStatus(graph.graph_status),
                message= "Execution of the graph",
                help = self.help,
                issue = self.issue,
                memory=memory,
                sub_queries=list(graph.sub_queries.values()),
                sub_query_need_attention=sub_query_need_attention,
                pasued_level=pasued_level,
            )

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                message=f"Execution failed: {str(e)}",
                help = self.help,
                issue = self.issue,
                memory=memory,
                sub_queries=list(graph.sub_queries.values()),
                sub_query_need_attention=sub_query_need_attention,
                pasued_level=pasued_level,
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
        **kwargs,
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
        classify_for_new_discussion = True,
        last_result: ExecutionResult = None,
        **kwargs,
    ) -> ExecutionResult:
        if classify_for_new_discussion:
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

        resume_from_level = last_result.pasued_level if last_result else 0
        sub_query_need_attention = last_result.sub_query_need_attention if last_result else None
        if not sub_query_need_attention:
            raise ValueError("No sub-query to resume from")
        
        if sub_query_need_attention.status == "human":
            sub_query_need_attention.internal_memory.append({"role": "assistant", "content": "\n".join(sub_query_need_attention.help)})
            sub_query_need_attention.internal_memory.append({"role": "user", "content": user_input})
        elif sub_query_need_attention.status in ["failed", "error"]:
            sub_query_need_attention.internal_memory.append({"role": "assistant", "content": "\n".join(sub_query_need_attention.issue)})
            sub_query_need_attention.internal_memory.append({"role": "user", "content": user_input})
        else:
            sub_query_need_attention.internal_memory.append({"role": "assistant", "content": "Please, help me understand what you mean, or provide more information."})
            sub_query_need_attention.internal_memory.append({"role": "user", "content": user_input})


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
        **kwargs,
    ) -> List[Dict[str, Any]]:
        original_sub_query = graph.sub_queries[index]
        results = []

        try:
            filtered_tools_info = {
                tool: info
                for tool, info in tools_info.items()
                if info["name"] == original_sub_query.tool
            } if original_sub_query.status == "pending" else tools_info

            message, usage = await tool_maker.make_tools(
                original_sub_query.task if original_sub_query.status == "pending" else None,
                filtered_tools_info, 
                context.get("memory", []) + original_sub_query.internal_memory
            )
            graph.update_token_usage(usage)
            
            if len(message["tool_calls"]) > 1:
                print(f"Multiple tool calls in a single message: {message}")
                

            sub_query = original_sub_query
            
            all_tools_results = []
            prev_name = sub_query.tool
            for tool_call in message["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                if tool_name != prev_name:
                    print(f"Multiple tools in a single message: {message}")
                    sub_query.other_tools.append(tool_name)
                tool_id = tool_call["id"]
                arguments = json.loads(tool_call["function"]["arguments"])

                if tool_name in tool_functions:
                    tool_result = {"tool_call_id": tool_id, "name": tool_name}
                    result = await tool_functions[tool_name](arguments, kwargs.get("extra", {}))
                    result_dict = json.loads(result) if type(result) == str else result
                    result_json = json.dumps(result_dict, indent=4)
                    tool_result["result"] = result
                    tool_result["help"] = result_dict.get("help", "")
                    tool_result["issue"] = result_dict.get("issue", "")
                    tool_result["status"] = result_dict.get("status", "success")
                    
                    
                    message_for_signle_tool_call = copy.deepcopy(message)
                    message_for_signle_tool_call['tool_calls'] = [tool_call]

                    memory_entry = [
                        # {"role": "user", "content": sub_query.task},
                        message_for_signle_tool_call,
                        {"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": result_json},
                    ]
                    tool_result["memory"] = memory_entry
                    
                    all_tools_results.append(tool_result)
                else:
                    raise ValueError(f"No function found for tool {tool_name}")

            from collections import Counter
            # Count all status
            status_counter = Counter([tool_result["status"] for tool_result in all_tools_results])
            
            if status_counter["success"] == len(all_tools_results):
                sub_query.status = "success"
            elif status_counter["failed"] == len(all_tools_results):
                sub_query.status = "failed"
            elif status_counter["human"] == len(all_tools_results):
                sub_query.status = "human"
            else:
                sub_query.status = "partial"
                
            sub_query.result = json.dumps([tool_result["result"] for tool_result in all_tools_results])
            
            sub_query.issue = [f"Tool {tool_result['name']}, Issue {ix}: {tool_result['issue']}" for ix, tool_result in enumerate(all_tools_results) if tool_result["issue"]]
            sub_query.help = [f"Tool {tool_result['name']}, Help {ix}: {tool_result['help']}" for ix, tool_result in enumerate(all_tools_results) if tool_result["help"]]

            sub_query.internal_memory = memory_entries = sum([tool_result["memory"] for tool_result in all_tools_results], [])
            sub_query.internal_memory = memory_entries = [{"role": "user", "content": sub_query.task}] + memory_entries
                            
            results.append({
                "index": index,
                "sub_query": sub_query,
                "status": sub_query.status,
                "memory": memory_entries,
            })

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