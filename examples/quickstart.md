# Tutorial: Handling Complex AI Queries with tool4ai

In this tutorial, we'll walk through the process of using the tool4ai library to handle complex queries, break them down into manageable sub-queries, execute them, and visualize the results. We'll also demonstrate how to incorporate human feedback into the execution process.

## 1. Setting Up the Environment

First, let's import the necessary modules and set up our environment:

```python
import asyncio
from typing import Dict, Any
from tool4ai.core.router import Router
from tool4ai.core.toolkit import Toolkit
from tool4ai.core.tool import Tool
from tool4ai.core.models import ExecutionResult, ExecutionStatus
from tool4ai.toolmakers.openai_maker import OpenAIToolMaker
from tool4ai.utils.config_manager import config_manager
```

## 2. Defining Tool Functions and Schemas

Now, let's define some tool functions and their corresponding schemas:

```python
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
    }
}

async def retrieve_favorites(arguments: Dict[str, Any]):
    list_name = arguments["list_name"]
    movies = db["favorite_lists"].get(list_name, [])
    return {
        "status": "success",
        "return": {"list_name": list_name, "movies": movies},
        "message": f"Retrieved {len(movies)} movies from {list_name}"
    }

async def recommend_similar_movies(arguments: Dict[str, Any]):
    movie = arguments["movie"]
    # In a real scenario, this would use a recommendation algorithm
    similar_movies = ["The Conjuring", "Insidious", "Sinister"]
    return {
        "status": "success",
        "return": {"similar_movies": similar_movies},
        "message": f"Found {len(similar_movies)} similar movies to {movie}"
    }

async def add_to_favorite(arguments: Dict[str, Any]):
    list_name = arguments["list_name"]
    movies = arguments["movies"]
    if list_name not in db["favorite_lists"]:
        db["favorite_lists"][list_name] = []
    db["favorite_lists"][list_name].extend(movies)
    return {
        "status": "success",
        "message": f"Added {len(movies)} movies to {list_name}",
        "return": {"added_movies": movies}
    }

async def check_true_story(arguments: Dict[str, Any]):
    movies = arguments["movies"]
    results = {}
    for movie in movies:
        based_on_true_story = db["movie_details"].get(movie, {}).get("based_on_true_story", "Unknown")
        results[movie] = based_on_true_story
    return {
        "status": "success",
        "return": {"results": results},
        "message": f"Checked {len(movies)} movies for true story basis"
    }

# Tool schemas
tool_schemas = {
    "retrieve_favorites": {
        "name": "retrieve_favorites",
        "description": "Retrieve movies from a favorite list",
        "parameters": {
            "type": "object",
            "properties": {
                "list_name": {"type": "string", "description": "The name of the favorite list to retrieve"}
            },
            "required": ["list_name"]
        }
    },
    "recommend_similar_movies": {
        "name": "recommend_similar_movies",
        "description": "Recommend movies similar to a given movie",
        "parameters": {
            "type": "object",
            "properties": {
                "movie": {"type": "string", "description": "The movie to find similar recommendations for"}
            },
            "required": ["movie"]
        }
    },
    "add_to_favorite": {
        "name": "add_to_favorite",
        "description": "Add movies to a favorite list",
        "parameters": {
            "type": "object",
            "properties": {
                "list_name": {"type": "string", "description": "The name of the favorite list"},
                "movies": {"type": "array", "items": {"type": "string"}, "description": "List of movie titles to add"}
            },
            "required": ["list_name", "movies"]
        }
    },
    "check_true_story": {
        "name": "check_true_story",
        "description": "Check if movies are based on true stories",
        "parameters": {
            "type": "object",
            "properties": {
                "movies": {"type": "array", "items": {"type": "string"}, "description": "List of movie titles to check"}
            },
            "required": ["movies"]
        }
    }
}
```

## 3. Creating the Toolkit

Now that we have our tool functions and schemas, let's create a Toolkit:

```python
def create_toolkit():
    toolkit = Toolkit()
    for name, schema in tool_schemas.items():
        tool = Tool(name=name, schema=schema["parameters"], description=schema["description"])
        toolkit.add_tool(tool)
    return toolkit

toolkit = create_toolkit()
```

## 4. Setting Up the Router

Next, we'll set up the Router with our Toolkit and a ToolMaker:

```python
tool_maker = OpenAIToolMaker(config_manager.get("llm.model"))
router = Router(toolkit, tool_maker)
```

## 5. Processing a Complex Query

Now, let's process our complex query:

```python
complex_query = "Find a horror movie from my 'Scary Nights' list, recommend similar movies, and add them to a new list called 'More Nightmares'. I can't wait to dive into these! Finally, let me know if any of these movies are based on true stories."

graph = router.route(complex_query)

print("Sub-queries generated:")
for i, sub_query in enumerate(graph.sub_queries.values()):
    print(f"{i+1}. {sub_query.task} (Tool: {sub_query.tool})")
```

## 6. Visualizing the Graph

To visualize the graph:

```python
# Generate a static PNG visualization
graph.visualize(output_file="complex_query_graph")
print("Static graph visualization saved as 'complex_query_graph.png'")

# Generate an interactive HTML visualization
graph.generate_interactive_html(output_file="complex_query_interactive.html")
print("Interactive graph visualization saved as 'complex_query_interactive.html'")

# If you need the graph data in Cytoscape JSON format
cytoscape_json = graph.to_cytoscape_json()
print("Cytoscape JSON data:", cytoscape_json)
```

## 7. Executing the Graph

Now, let's execute the graph and handle human feedback:

```python
async def execute_graph(graph, tool_functions, max_iterations=5):
    context = {"memory": []}
    iteration = 0
    
    while iteration < max_iterations:
        result = await graph.execute(tool_functions, toolkit.to_json_schema(), context, router.tool_maker, add_human_failed_memory=True)
        
        if result.status == ExecutionStatus.SUCCESS:
            print("Execution completed successfully!")
            return result
        elif result.status == ExecutionStatus.HUMAN:
            print("Human input required:")
            print(result.message)
            user_input = input("Please provide your input: ")
            result = await graph.resume_execution(user_input, tool_functions, toolkit.to_json_schema(), context, router.tool_maker)
        else:
            print(f"Execution failed: {result.message}")
            return result
        
        iteration += 1
    
    print("Maximum iterations reached. Execution incomplete.")
    return result

tool_functions = {
    "retrieve_favorites": retrieve_favorites,
    "recommend_similar_movies": recommend_similar_movies,
    "add_to_favorite": add_to_favorite,
    "check_true_story": check_true_story
}

final_result = asyncio.run(execute_graph(graph, tool_functions))

print("\nFinal Execution Results:")
for sub_query in final_result.sub_queries:
    print(f"Task: {sub_query.task}")
    print(f"Tool: {sub_query.tool}")
    print(f"Status: {sub_query.status}")
    print(f"Result: {sub_query.result}")
    print("---")
```

This tutorial demonstrates how to use the tool4ai library to handle complex queries, break them down into manageable sub-queries, execute them, and incorporate human feedback when necessary. The process includes:

1. Setting up the environment and importing necessary modules
2. Defining tool functions and their schemas
3. Creating a Toolkit with the defined tools
4. Setting up a Router with the Toolkit and a ToolMaker
5. Processing a complex query and generating sub-queries
6. Visualizing the execution graph
7. Executing the graph, handling human feedback, and displaying the results

By following this tutorial, you can create a powerful system for handling complex AI queries, breaking them down into smaller, manageable tasks, and executing them with the ability to incorporate human feedback when needed.