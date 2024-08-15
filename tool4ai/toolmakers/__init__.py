from .tool_convertors import ToolsConvertor, OpenAIToolConvertor, AnthropicToolConvertor

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import litellm, json

class ToolMaker(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.system_prompt = "You are an AI assistant designed to analyze user queries and determine which tools, if any, should be used to respond."

    @abstractmethod
    def extract_usage(self, response) -> Dict[str, int]:
        pass

    @abstractmethod
    async def make_tools(self, query: str, tools_info: Dict[str, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _create_messages(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass
    
    async def completion(self, 
                         system_message: str, 
                         user_prompt: str, 
                         json_schema: Optional[Dict[str, Any]] = None,
                         **kwargs) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Perform an asynchronous completion call to the language model.

        Args:
            system_message (str): The system message to set the context.
            user_prompt (str): The user's prompt or query.
            json_schema (Optional[Dict[str, Any]]): JSON schema for response formatting.
            **kwargs: Additional keyword arguments for model parameters.

        Returns:
            Tuple[Dict[str, Any], Dict[str, int]]: The completion response and usage statistics.
        """
        default_params = {
            "temperature": 0.7,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }

        # Update default parameters with any provided kwargs
        params = {**default_params, **kwargs}

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]

        if json_schema:
            json_schema["schema"]["additionalProperties"] = False
            json_schema["strict"] = True
            
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema
            }

        try:
            response = await litellm.acompletion(
                model=self.model_name,
                messages=messages,
                **params
            )

            message = response.choices[0].message.to_dict()
            
            usage = response.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

            return message, usage

        except Exception as e:
            print(f"Error in completion: {str(e)}")
            raise

    async def chat(self, 
                         messages: List[Dict[str, Any]],
                         **kwargs) -> Tuple[Dict[str, Any], Dict[str, int]]:
        default_params = {
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }

        # Update default parameters with any provided kwargs
        params = {**default_params, **kwargs}

        try:
            response = await litellm.acompletion(
                model=self.model_name,
                messages=messages,
                **params
            )

            message = response.choices[0].message.to_dict()
            
            usage = response.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

            return message, usage

        except Exception as e:
            print(f"Error in completion: {str(e)}")
            raise


# Define what should be importable from the package
__all__ = ['ToolsConvertor', 'OpenAIToolConvertor', 'AnthropicToolConvertor', 'ToolMaker']