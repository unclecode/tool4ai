# result_generator.py
import json
from typing import List, Dict, Any
from ..models import SubQuery
from ...toolmakers import ToolMaker

class ResultGenerator:
    async def generate_interim_message(self, tool_maker: ToolMaker, level_results: List[SubQuery], context: Dict[str, Any]) -> str:
        system_message = """You are an AI assistant engaged in a chat conversation with a user. Your task is to generate a concise, friendly, and personalized message about the current state of their request execution. Focus on what's immediately relevant for the user to proceed.

        Guidelines:
        1. Be brief and to the point.
        2. Personalize the message using the user's profile information if available.
        3. If human input is needed, clearly state what information is required.
        4. If errors occurred, briefly mention them and provide a simple suggestion if possible.
        5. Use a casual and friendly tone.
        6. Summarize the issues, don't list all tasks or provide excessive details.
        7. This is mid-conversation, so don't use greetings like "Hi" or "Hey".
        8. If the user's name is available in context['user_profile']['name'], use it; otherwise, don't use any name.
        9. Try to find quick and alternative solutions to the problems faced by the user."""

        user_profile = context.get("user_profile", {})
        user_name = user_profile.get("name")

        human_input_needed = [sq for sq in level_results if sq.status == "human"]
        errors_occurred = [sq for sq in level_results if sq.status == "failed"]

        human_input_summary = "\n".join([f"- {sq.issue}" for sq in human_input_needed])
        error_summary = "\n".join([f"- {sq.issue}" for sq in errors_occurred])

        user_prompt = f"""Generate a brief, friendly message for the user about their request. Here's the current situation:

        User's name: {user_name if user_name else "Not provided"}
        Tasks needing human input: {len(human_input_needed)}
        Tasks with errors: {len(errors_occurred)}

        Human input needed:
        {human_input_summary}

        Errors occurred:
        {error_summary}

        If human input is needed, briefly state what's required. If errors occurred, briefly mention them with a simple suggestion. Keep your response short, friendly, and focused on what the user needs to do next. If there are more than 3 issues in either category, mention that there are additional issues not listed. Remember, this is mid-conversation, so don't use greetings."""

        json_schema = {
            "name": "response",
            "schema": {
                "type": "object",
                "properties": {"reply": {"type": "string"}},
                "required": ["reply"],
            },
        }

        result, usage = await tool_maker.completion(system_message, user_prompt, json_schema)
        reply = json.loads(result["content"]).get("reply")
        return reply, usage

    async def generate_final_response(self, context: Dict[str, Any], tool_maker: ToolMaker, final_prompt: str) -> str:
        context["memory"].append({"role": "user", "content": final_prompt})
        message, usage = await tool_maker.chat(context["memory"])
        return message, usage

    async def classify_user_input(self, tool_maker: Any, user_query: str, context: Dict[str, Any]) -> str:
        system_message = """You are an AI assistant integrated into a movie recommendation and list management system. Your task is to classify whether a user's latest message is starting a new discussion or responding to a previously paused task execution.

        Guidelines:
        1. Analyze the user's latest query in the context of the conversation history.
        2. Look for clear indicators of a new topic, direct responses to previous questions, or decisions to abandon the current task.
        3. Be sensitive to context and implicit references to previous issues.
        4. If in doubt, lean towards classifying as a continuation of the previous discussion.

        Classification:
        - "new_discussion": The user is clearly starting a new topic, request, or explicitly abandoning the current task.
        - "continuation": The user is responding to or addressing a previous issue or request."""

        user_prompt = f"""Classify the following user input as either a new discussion or a continuation of the previous interaction. Here's the relevant information:

        Recent conversation history (last 5 exchanges):
        {context['memory'][-5:]}

        User's latest query: "{user_query}"

        Provide your classification as either "new_discussion" or "continuation"."""

        json_schema = {
            "name": "response",
            "schema": {
                "type": "object",
                "properties": {
                    "classification": {
                        "type": "string",
                        "enum": ["new_discussion", "continuation"]
                    }
                },
                "required": ["classification"]
            }
        }

        result, usage = await tool_maker.completion(system_message, user_prompt, json_schema, max_tokens=50)
        classification = json.loads(result["content"]).get("classification")
        return classification, usage 
