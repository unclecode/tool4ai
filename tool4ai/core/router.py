# File: tool4ai/core/router.py
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from ..core.toolkit import Toolkit
from ..utils.config_manager import config_manager
from ..toolmakers import ToolMaker
from ..core.models import SubQuery, SubQueryResponse
from ..toolmakers.openai_maker import OpenAIToolMaker
from .graph.tool_dependency_graph import ToolDependencyGraph
from .models import SubQuery, SubQueryResponse
import litellm
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed

SYS_MESSAGE = """Your goal is to break down a complex user query into smaller, actionable sub-queries and rewrite each in a way that is clear and suitable for an AI agent to take action. 

**Each sub-query must be atomic, meaning that if it requires a tool, it should only need one tool.** 

If a sub-query would require multiple tools, it must be further broken down into smaller sub-queries, each associated with only one tool. If the query is simple, return it as is but still rewrite it in the 'task' field for clarity. The user may provide a list of available tools, but not every sub-query will require a tool. The expected output format is a JSON array with each item containing the following fields:

1. **'index'**: The index of the sub-query in the sequence (starting from 0).
2. **'sub_query'**: The original sub-query.
3. **'task'**: The rewritten version suitable for AI action.
4. **'tool'**: The name of the tool to use, if applicable. Empty string if sub-query is not actionable.
5. **'dependent_on'**: The index of the sub-query that this sub-query is dependent on. Set to -1 if there is no dependency or sub-query is not actionable.
6. **'dependency_attr'**: The name of the attribute in the tool that is dependent on the previous sub-query. Empty string if sub-query is not actionable.

Ensure that the sub-queries are ordered as follows: 
1. First, list all actionable sub-queries that are not dependent on any other sub-query.
2. Then, list actionable sub-queries that are dependent on previous sub-queries.
3. Finally, place all non-actionable sub-queries at the end.

The 'dependent_on' field should contain the index of the sub-query that this sub-query depends on, or -1 if there is no dependency. The 'dependency_attr' field should be a string representing the attribute in the tool that depends on the output of the previous sub-query.

### Example:
**Complex Query:** Find a sci-fi movie from my 'Sci-Fi Favorites' list, recommend similar movies, add those recommendations to a new 'Sci-Fi Discoveries' list, check the director of the first recommended movie and find more movies by them, and by the way, I'm feeling nostalgic about these classic sci-fi films.

**Expected Output:**
[
    {
        "index": 0,
        "sub_query": "Find a sci-fi movie from my 'Sci-Fi Favorites' list.",
        "task": "Retrieve a sci-fi movie from the 'Sci-Fi Favorites' list.",
        "tool": "retrieve_favorites",
        "dependent_on": -1,
        "dependency_attr": ""
    },
    {
        "index": 1,
        "sub_query": "Recommend similar movies.",
        "task": "Recommend movies similar to the one retrieved from the 'Sci-Fi Favorites' list.",
        "tool": "recommend_similar_movies",
        "dependent_on": 0,
        "dependency_attr": "movie_title"
    },
    {
        "index": 2,
        "sub_query": "Create a new 'Sci-Fi Discoveries' list.",
        "task": "Create a new favorite list with the title 'Sci-Fi Discoveries'.",
        "tool": "create_favorite_list",
        "dependent_on": -1,
        "dependency_attr": ""
    },
    {
        "index": 3,
        "sub_query": "Add those recommendations to the 'Sci-Fi Discoveries' list.",
        "task": "Add the recommended movies to the 'Sci-Fi Discoveries' list.",
        "tool": "add_to_favorite",
        "dependent_on": 1,
        "dependency_attr": "movies"
    },
    {
        "index": 4,
        "sub_query": "Check the director of the first recommended movie and find more movies by them.",
        "task": "Retrieve information about the director of the first recommended movie and find more movies by that director.",
        "tool": "get_director_info",
        "dependent_on": 1,
        "dependency_attr": "movie_title"
    },
    {
        "index": 5,
        "sub_query": "By the way, I'm feeling nostalgic about these classic sci-fi films.",
        "task": "Acknowledge the user's sentiment about feeling nostalgic. No action is required.",
        "tool": "",
        "dependent_on": -1,
        "dependency_attr": ""
    }
]

### Key Emphasis:
1. **Atomic Sub-queries**: Each sub-query must be atomic and require only one tool. Break down sub-queries further if they require multiple tools. For example ven in the case of "Execut A, and B in parallel" you should break this down into two sub-queries.
2. **Ordering**: Start with independent actionable sub-queries, followed by dependent actionable sub-queries, and end with non-actionable sub-queries.
3. **Indexing**: Assign an index to each sub-query in the order they should be executed.
4. **Dependencies**: Use the 'dependent_on' field to indicate dependencies by referencing the index of the dependent sub-query.
5. **Dependency Attribute**: Specify the exact attribute in the tool that depends on the output of the previous sub-query in the 'dependency_attr' field.
6. **Efficiency of Dependency**: Make sure to assign dependency in the way that total execution of the detected tools become minimal. More parallel call ends to faster execution"""


class Router:
    def __init__(self, toolkit: Toolkit, tool_maker: ToolMaker = None):
        if not isinstance(toolkit, Toolkit):
            raise TypeError("toolkit must be an instance of Toolkit")
        self.toolkit = toolkit
        # self.tool_maker = tool_maker or ToolMaker(config_manager.get("llm.model"))
        self.tool_maker = tool_maker or OpenAIToolMaker(
            config_manager.get("llm.model")
        )
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _update_token_usage(self, usage):
        for key in self.token_usage:
            self.token_usage[key] += usage.get(key, 0)

    def get_total_token_usage(self):
        return self.token_usage

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def gen_subquery(self, query: str, toolkit: Toolkit = None) -> SubQueryResponse:
        tools = toolkit.to_markdown() if toolkit else ""
        try:
            response = litellm.completion(
                model="gpt-4o-mini-2024-07-18",
                # model="gpt-4o-2024-08-06",
                # model="ft:gpt-4o-mini-2024-07-18:kidocode:sub-querizer:9uxxtTXq:ckpt-step-222",
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": SYS_MESSAGE}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"<tools>{tools}</tools>\nComplex query: {query}",
                            }
                        ],
                    },
                ],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "sub_query_response",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "sub_queries": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "index": {
                                                "type": "integer",
                                                "description": "The index of the sub-query in the sequence.",
                                            },
                                            "sub_query": {
                                                "type": "string",
                                                "description": "The original sub-query from the user's complex query.",
                                            },
                                            "task": {
                                                "type": "string",
                                                "description": "The rewritten version of the sub-query suitable for AI action.",
                                            },
                                            "tool": {
                                                "type": "string",
                                                "description": "The name of the tool to use for this sub-query, if applicable.",
                                            },
                                            "dependent_on": {
                                                "type": "integer",
                                                "description": "The index of the sub-query that this sub-query is dependent on, or -1 if there is no dependency.",
                                            },
                                            "dependency_attr": {
                                                "type": "string",
                                                "description": "The name of the attribute in the tool that depends on the output of the previous sub-query.",
                                            },
                                        },
                                        "required": [
                                            "index",
                                            "sub_query",
                                            "task",
                                            "tool",
                                            "dependent_on",
                                            "dependency_attr",
                                        ],
                                        "additionalProperties": False,
                                    },
                                }
                            },
                            "required": ["sub_queries"],
                            "additionalProperties": False,
                        },
                    },
                },
            )
            message = response.choices[0].message.to_dict()
            usage = response.get(
                "usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
            content = message["content"]
            result = SubQueryResponse.model_validate_json(content)
            return result, usage
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {str(e)}")
            raise
        except Exception as e:
            print(f"Error in gen_subquery: {str(e)}")
            raise

    def route(self, query: str, context: Dict[str, Any] = None) -> ToolDependencyGraph:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if context is not None and not isinstance(context, dict):
            raise TypeError("context must be a dictionary or None")

        context = context or {}

        # Generate sub-queries
        sub_query_response, usage = self.gen_subquery(query, self.toolkit)
        # Add this after parsing the response
        for sub_query in sub_query_response.sub_queries:
            assert isinstance(
                sub_query.index, int
            ), f"Index must be an integer, got {type(sub_query.index)}"
            assert isinstance(
                sub_query.dependent_on, int
            ), f"dependent_on must be an integer, got {type(sub_query.dependent_on)}"
            assert isinstance(
                sub_query.dependency_attr, str
            ), f"dependency_attr must be a string, got {type(sub_query.dependency_attr)}"

        self._update_token_usage(usage)

        # Create ToolDependencyGraph
        toolsDepGraph = ToolDependencyGraph()
        toolsDepGraph.build_dependency_structure(sub_query_response)

        return toolsDepGraph

    def __repr__(self) -> str:
        return f"Router(toolkit={self.toolkit}, tool_maker={self.tool_maker})"


# Example usage
def main():
    # Initialize your toolkit and ToolMaker here
    toolkit = Toolkit()
    tool_maker = ToolMaker(config_manager.get("llm.model"))

    router = Router(toolkit, tool_maker)
    result = router.route("Your complex query here")

    print("Sub-queries and their results:")
    for sq, res in result["sub_queries"]["executed"].items():
        print(f"  {sq.task}: {res}")

    print("\nNon-actionable sub-queries:")
    for sq in result["sub_queries"]["non_actionable"]:
        print(f"  {sq['task']}")

    print("\nDependency Graph:")
    print(result["dependency_graph"])


if __name__ == "__main__":
    main()
