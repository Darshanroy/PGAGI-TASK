from typing import List, Optional
import requests

from langchain.schema import AIMessage, SystemMessage

from agents.agent_simulations.agent.dialogue_agent import DialogueAgent
from agents.conversational.output_parser import ConvoOutputParser
from config import Config
from memory.zep.zep_memory import ZepMemory
from services.run_log import RunLogsManager
from typings.agent import AgentWithConfigsOutput

class DialogueAgentWithTools(DialogueAgent):
    def __init__(
        self,
        name: str,
        agent_with_configs: AgentWithConfigsOutput,
        system_message: SystemMessage,
        model: str,  # Changed to string to specify the model name directly
        tools: List[any],
        session_id: str,
        sender_name: str,
        is_memory: bool = False,
        run_logs_manager: Optional[RunLogsManager] = None,
        **tool_kwargs,
    ) -> None:
        super().__init__(name, agent_with_configs, system_message, model)
        self.tools = tools
        self.session_id = session_id
        self.sender_name = sender_name
        self.is_memory = is_memory
        self.run_logs_manager = run_logs_manager

    def send(self) -> str:
        """
        Applies the chat model to the message history
        and returns the message string
        """

        memory: ZepMemory

        if self.is_memory:
            memory = ZepMemory(
                session_id=self.session_id,
                url=Config.ZEP_API_URL,
                api_key=Config.ZEP_API_KEY,
                memory_key="chat_history",
                return_messages=True,
            )
            memory.human_name = self.sender_name
            memory.ai_name = self.agent_with_configs.agent.name
            memory.auto_save = False

        prompt = "\n".join(self.message_history + [self.prefix])

        # Assuming XAgent runs as a separate service and we need to call it via HTTP
        xagent_url = "http://localhost:8090/api/run"  # URL of the running XAgent service
        headers = {"Content-Type": "application/json"}
        payload = {
            "task": prompt,
            "model": self.model,  # Assuming model is a string specifying the model name
            "config_file": "assets/config.yml"
        }

        try:
            response = requests.post(xagent_url, json=payload, headers=headers)
            response.raise_for_status()
            res = response.json().get("response", "No response from XAgent service.")
        except requests.RequestException as e:
            res = f"Error communicating with XAgent service: {str(e)}"

        # Optionally, save the AI message to memory if needed
        # if self.is_memory:
        #     memory.save_ai_message(res)

        message = AIMessage(content=res)

        return message.content
