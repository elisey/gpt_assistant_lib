from ._assistant import Assistant, AssistantInterface
from ._history import HistoryInterface, SimpleHistory
from ._openai_client import OpenAI


def build_assistant(openai_api_key: str, prompt: str, history_size: int, history_ttl: int) -> AssistantInterface:
    def build_history() -> HistoryInterface:
        history = SimpleHistory(history_size, history_ttl)
        history.init_system_content(prompt)
        return history

    openai_client = OpenAI(openai_api_key)

    return Assistant(openai_client, build_history)
