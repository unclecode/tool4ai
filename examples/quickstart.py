import os, sys

# append the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# Tutorial: Handling Complex AI Queries with tool4ai
## 1. Setting Up the Environment

import asyncio
from typing import Dict, Any
from tool4ai.core.router import Router
from tool4ai.core.toolkit import Toolkit
from tool4ai.core.tool import Tool
from tool4ai.core.models import ExecutionResult, ExecutionStatus
from tool4ai.toolmakers.openai_maker import OpenAIToolMaker
from tool4ai.utils.config_manager import config_manager
from tool4ai.core.graph.tool_dependency_graph import ToolDependencyGraph

## 2. Defining Tool Functions and Schemas

# Mock database for testing
db = {
    "favorite_lists": {
        "Scary Nights": ["The Shining", "Halloween", "A Nightmare on Elm Street"],
    },
    "movies": {
        "horror": {
            2020: ["The Invisible Man", "Host"],
            2021: ["A Quiet Place Part II", "Malignant"],
        },
    },
    "movie_details": {
        "The Shining": {"based_on_true_story": False},
        "The Conjuring": {"based_on_true_story": True},
    },
}


async def retrieve_favorites(arguments: Dict[str, Any]):
    list_name = arguments["list_name"]
    movies = db["favorite_lists"].get(list_name, [])
    return {
        "status": "success",
        "return": {"list_name": list_name, "movies": movies},
    }


async def recommend_similar_movies(arguments: Dict[str, Any]):
    movie = arguments["movie"]
    # In a real scenario, this would use a recommendation algorithm
    similar_movies = ["The Conjuring", "Insidious", "Sinister"]
    return {
        "status": "success",
        "return": {"similar_movies": similar_movies},
    }

help_count = 1
async def add_to_favorite(arguments: Dict[str, Any]):
    global help_count
    list_name = arguments["list_name"]
    movies = arguments["movies"]
    if help_count > 0 or list_name not in db["favorite_lists"]:
        help_count -= 1
        db["favorite_lists"][list_name] = []
        return {
            "status": "human",
            "help": f"List '{list_name}' does not exist. Do you want to create it?",
            "return": {"list_name": list_name, "movies": movies},
        }

    db["favorite_lists"][list_name].extend(movies)
    return {
        "status": "success",
        "return": {"added_movies": movies},
    }

# Create a tool for create favorite list
async def create_favorite_list(arguments: Dict[str, Any]):
    list_name = arguments["list_name"]
    if list_name in db["favorite_lists"]:
        return {
            "status": "success",
            "help": f"List '{list_name}' already exists.",
            "return": {"list_name": list_name},
        }

    db["favorite_lists"][list_name] = []
    return {
        "status": "success",
        "return": {"list_name": list_name},
    }

async def check_true_story(arguments: Dict[str, Any]):
    movies = arguments["movies"]
    results = {}
    for movie in movies:
        based_on_true_story = (
            db["movie_details"].get(movie, {}).get("based_on_true_story", "Unknown")
        )
        results[movie] = based_on_true_story
    return {
        "status": "success",
        "return": {"results": results},
    }


# Tool schemas
tool_schemas = {
    "retrieve_favorites": {
        "name": "retrieve_favorites",
        "description": "Retrieve movies from a favorite list",
        "parameters": {
            "type": "object",
            "properties": {
                "list_name": {
                    "type": "string",
                    "description": "The name of the favorite list to retrieve",
                }
            },
            "required": ["list_name"],
        },
    },
    "recommend_similar_movies": {
        "name": "recommend_similar_movies",
        "description": "Recommend movies similar to a given movie",
        "parameters": {
            "type": "object",
            "properties": {
                "movie": {
                    "type": "string",
                    "description": "The movie to find similar recommendations for",
                }
            },
            "required": ["movie"],
        },
    },
    'create_favorite_list': {
        'name': 'create_favorite_list',
        'description': 'Create a new favorite list',
        'parameters': {
            'type': 'object',
            'properties': {
                'list_name': {
                    'type': 'string',
                    'description': 'The name of the favorite list to create',
                }
            },
            'required': ['list_name']
        }
    },
    "add_to_favorite": {
        "name": "add_to_favorite",
        "description": "Add movies to a favorite list",
        "parameters": {
            "type": "object",
            "properties": {
                "list_name": {
                    "type": "string",
                    "description": "The name of the favorite list",
                },
                "movies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of movie titles to add",
                },
            },
            "required": ["list_name", "movies"],
        },
    },
    "check_true_story": {
        "name": "check_true_story",
        "description": "Check if movies are based on true stories",
        "parameters": {
            "type": "object",
            "properties": {
                "movies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of movie titles to check",
                }
            },
            "required": ["movies"],
        },
    },
}

tool_functions = {
    "retrieve_favorites": retrieve_favorites,
    "recommend_similar_movies": recommend_similar_movies,
    "add_to_favorite": add_to_favorite,
    "check_true_story": check_true_story,
    'create_favorite_list': create_favorite_list
}


## 3. Creating the Toolkit
def create_toolkit():
    toolkit = Toolkit()
    for name, schema in tool_schemas.items():
        tool = Tool(
            name=name,
            schema=schema["parameters"],
            description=schema["description"],
            f=tool_functions[name],
        )
        toolkit.add_tool(tool)
    return toolkit


toolkit = create_toolkit()

## 4. Setting Up the Router
tool_maker = OpenAIToolMaker(config_manager.get("llm.model"))
router = Router(toolkit, tool_maker)

## 5. Processing a Complex Query
complex_query = "Find a horror movie from my 'Scary Nights' list, recommend similar movies, and add them to a new list called 'More Nightmares'. I can't wait to dive into these! Finally, let me know if any of these movies are based on true stories."

graph = router.route(complex_query)
# # b24dc997-ba55-4fb8-8e39-6cd2f8feb600
# print("Run Id", graph.run_id)
# graph.save_sync()
# graph = ToolDependencyGraph.load_sync(graph.run_id)

graph: ToolDependencyGraph = ToolDependencyGraph.load_sync(
    "b24dc997-ba55-4fb8-8e39-6cd2f8feb600"
)


print("Sub-queries generated:")
for i, sub_query in enumerate(graph.sub_queries.values()):
    print(f"{i+1}. {sub_query.task} (Tool: {sub_query.tool})")

## 6. Visualizing the Graph
# Generate a static PNG visualization
# graph.visualize(output_file=os.path.join(__location__, "complex_query_graph"))
print("Static graph visualization saved as 'complex_query_graph.png'")

# Generate an interactive HTML visualization
# graph.generate_interactive_html(output_file=os.path.join(__location__, "complex_query_interactive.html"))
print("Interactive graph visualization saved as 'complex_query_interactive.html'")

# If you need the graph data in Cytoscape JSON format
cytoscape_json = graph.to_cytoscape_json()
print("Cytoscape JSON data:", cytoscape_json)


## 7. Executing the Graph
async def execute_graph(graph, toolkit, max_iterations=5):
    context = {"memory": []}
    iteration = 0

    while iteration < max_iterations:
        # result = await graph.execute(tool_functions, toolkit.to_json_schema(), context, router.tool_maker, add_human_failed_memory=True)
        result = await graph.execute(
            toolkit=toolkit,
            context=context,
            tool_maker=router.tool_maker,
            add_human_failed_memory=True,
        )

        if result.status == ExecutionStatus.SUCCESS:
            print("Execution completed successfully!")
            return result
        elif result.status == ExecutionStatus.HUMAN:
            print("Human input required:")
            print(result.help)
            user_input = "yes, pleas" # input("Please provide your input: ")
            result = await graph.resume_execution(
                user_input=user_input,
                toolkit=toolkit,
                context=context,
                tool_maker=router.tool_maker,
                classify_for_new_discussion=False,
                last_result = result
            )
            if result.status == ExecutionStatus.SUCCESS:
                print("Execution completed successfully!")
                return result
        else:
            print(f"Execution failed: {result.message}")
            return result

        iteration += 1

    print("Maximum iterations reached. Execution incomplete.")
    return result


final_result = asyncio.run(execute_graph(graph, toolkit))

print("\nFinal Execution Results:")
print("Message:", final_result.message)
print("Help:", final_result.help)
print("Issue:", final_result.issue)

for sub_query in final_result.sub_queries:
    print(f"Task: {sub_query.task}")
    print(f"Tool: {sub_query.tool}")
    print(f"Status: {sub_query.status}")
    print(f"Result: {sub_query.result}")
    print(f"Help: {sub_query.help}")
    print(f"Issue: {sub_query.issue}")
    print("---")
