# tests/functional/test_tool4ai_scenarios.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

import pytest
import json
from typing import Dict, Any

from tool4ai.core.router import Router
from tool4ai.core.toolkit import Toolkit
from tool4ai.core.tool import Tool
from tool4ai.core.models import ExecutionResult, ExecutionStatus
from tool4ai.toolmakers.openai_maker import OpenAIToolMaker
from tool4ai.utils.config_manager import config_manager

# Mock database for testing
db = {
    "favorite_lists": {
        "action movies": ["Die Hard", "Mad Max: Fury Road"],
        "sci-fi favorites": ["The Matrix", "Inception"],
    },
    "movies": {
        "action": {2020: ["Tenet", "Extraction"]},
        "sci-fi": {2021: ["Dune", "The Tomorrow War"]},
        "comedy": {2021: ["Free Guy", "Don't Look Up"]},
    },
}


# Tool functions
async def add_to_favorite(arguments: Dict[str, Any]):
    list_name = arguments["list_name"].lower()
    movies = arguments["movies"]
    if list_name not in db["favorite_lists"]:
        db["favorite_lists"][list_name] = []
    db["favorite_lists"][list_name].extend(movies)
    return json.dumps(
        {
            "status": "success",
            "message": f"Added {len(movies)} movies to {list_name}",
            "return": {
                "message": f"Added {len(movies)} movies to {list_name}",
                "added_movies": movies,
            },
        }
    )


async def retrieve_favorites(arguments: Dict[str, Any]):
    list_name = arguments["list_name"].lower()
    movies = db["favorite_lists"].get(list_name, [])
    return json.dumps(
        {
            "status": "success",
            "return": {"list_name": list_name, "movies": movies},
            "message": f"Retrieved {len(movies)} movies from {list_name}",
        }
    )


async def search_movies(arguments: Dict[str, Any]):
    genre = arguments["genre"].lower()
    year = arguments.get("year")
    movies = db["movies"].get(genre, {}).get(year, [])
    if not movies:
        return json.dumps(
            {
                "status": "human",
                "message": f"No movies found for {genre} in {year}. Would you like to search for a different year?",
                "return": {
                    "genre": genre,
                    "year": year,
                    "movies": movies,
                },
            }
        )
    return json.dumps(
        {
            "status": "success",
            "return": {"genre": genre, "year": year, "movies": movies},
            "message": f"Found {len(movies)} movies in {genre} for {year}",
        }
    )


# Tool schemas
tool_schemas = {
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
    "search_movies": {
        "name": "search_movies",
        "description": "Search for movies by genre and year",
        "parameters": {
            "type": "object",
            "properties": {
                "genre": {
                    "type": "string",
                    "description": "The genre of movies to search for",
                },
                "year": {
                    "type": "integer",
                    "description": "The year of movies to search for",
                },
            },
            "required": ["genre", "year"],
        },
    },
}


# Fixtures
@pytest.fixture(scope="module")
def sample_toolkit():
    toolkit = Toolkit()
    for name, schema in tool_schemas.items():
        tool = Tool(
            name=name, schema=schema["parameters"], description=schema["description"]
        )
        toolkit.add_tool(tool)
    return toolkit


@pytest.fixture(scope="module")
def router(sample_toolkit):
    tool_maker = OpenAIToolMaker(config_manager.get("llm.model"))
    return Router(sample_toolkit, tool_maker)


# 3.1 End-to-End Scenarios
@pytest.mark.asyncio
async def test_end_to_end_scenario(router):
    query = "Find sci-fi movies from 2021 and add them to my 'sci-fi favorites' list"
    graph = router.route(query)

    tool_functions = {
        "search_movies": search_movies,
        "add_to_favorite": add_to_favorite,
        "retrieve_favorites": retrieve_favorites,
    }

    result = await graph.execute(tool_functions, router.toolkit.to_json_schema(), {}, router.tool_maker)

    assert result.status == ExecutionStatus.SUCCESS
    assert "Dune" in db["favorite_lists"]["sci-fi favorites"]
    assert "The Tomorrow War" in db["favorite_lists"]["sci-fi favorites"]

# 3.2 Error Handling and Edge Cases
@pytest.mark.asyncio
async def test_invalid_query(router):
    query = "This is an invalid query that doesn't relate to movies"
    graph = router.route(query)

    tool_functions = {
        "search_movies": search_movies,
        "add_to_favorite": add_to_favorite,
        "retrieve_favorites": retrieve_favorites,
    }

    result = await graph.execute(tool_functions, router.toolkit.to_json_schema(), {}, router.tool_maker)

    assert result.status == ExecutionStatus.FAILED or len(graph.non_actionable_sub_queries) == 1

@pytest.mark.asyncio
async def test_missing_tool(router):
    query = "Find action movies from 2020 and add them to my 'action movies' list"
    graph = router.route(query)

    tool_functions = {
        "search_movies": search_movies,
        # Intentionally omitting add_to_favorite
    }

    result = await graph.execute(tool_functions, router.toolkit.to_json_schema(), {}, router.tool_maker)

    assert result.status == ExecutionStatus.FAILED
    assert any(sq.status == "failed" for sq in result.sub_queries if sq.tool == "add_to_favorite")


# 3.3 Human Interaction Scenarios
@pytest.mark.asyncio
async def test_human_interaction_mid_execution(router):
    query = "Find comedy movies from 2022 and add them to my 'comedy favorites' list"
    graph = router.route(query)
    
    tool_functions = {
        "search_movies": search_movies,
        "add_to_favorite": add_to_favorite,
        "retrieve_favorites": retrieve_favorites,
    }
    
    result = await graph.execute(tool_functions, router.toolkit.to_json_schema(), {}, router.tool_maker)
    
    assert result.status == ExecutionStatus.HUMAN
    assert any(sq.status == "human" for sq in result.sub_queries)
    assert "No movies found for comedy in 2022" in [sq.result for sq in result.sub_queries if sq.status == "human"][0]

@pytest.mark.asyncio
async def test_resume_execution_after_human_input(router):
    # Initial query
    query = "Find comedy movies from 2022 and add them to my 'comedy favorites' list"
    graph = router.route(query)
    
    tool_functions = {
        "search_movies": search_movies,
        "add_to_favorite": add_to_favorite,
        "retrieve_favorites": retrieve_favorites,
    }

    context = {
        "memory": []
    }
    # First execution (will require human input)
    result = await graph.execute(tool_functions, router.toolkit.to_json_schema(), context, router.tool_maker, add_human_failed_memory=True)
    assert result.status == ExecutionStatus.HUMAN    
    
    # Simulate human input
    user_input = "Yes, please search for comedy movies from 2021 instead."
    result = await graph.resume_execution(user_input, tool_functions, router.toolkit.to_json_schema(), context, router.tool_maker)    
    
    assert result.status == ExecutionStatus.SUCCESS
    assert any(sq.tool == "search_movies" and "2021" in sq.result for sq in result.sub_queries)
    assert any(sq.tool == "add_to_favorite" and "comedy favorites" in sq.result for sq in result.sub_queries)



if __name__ == "__main__":
    pytest.main([__file__])
