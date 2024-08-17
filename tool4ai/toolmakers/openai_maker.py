from .tool_convertors import OpenAIToolConvertor
from ..toolmakers import ToolMaker
from .tool_convertors import OpenAIToolConvertor
from typing import Dict, Any, List, Tuple
import json
import litellm

class OpenAIToolMaker(ToolMaker):
    def __init__(self, model_name: str = "gpt-4o-mini-2024-07-18"):
        super().__init__(model_name)
        self.tool_convertor = OpenAIToolConvertor()

    def extract_usage(self, response) -> Dict[str, int]:
        return response.get('usage', {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        })
    
    async def make_tools(self, query: str, tools_info: Dict[str, Dict[str, Any]], memory: List[Dict[str, Any]]) -> Dict[str, Any]:
        messages = self._create_messages(query, memory)
        tools = self.tool_convertor.convert(list(tools_info.values()))

        try:
            response = await litellm.acompletion(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="required",
            )
            message = response.choices[0].message.to_dict()
            usage = self.extract_usage(response)
            return message, usage
        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            raise

    def _create_messages(self, query: str, memory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(memory)
        if query:
            messages.append({"role": "user", "content": query})
        return messages
