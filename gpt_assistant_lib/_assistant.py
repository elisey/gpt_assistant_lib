import abc
from typing import Callable

from ._history import HistoryInterface, Role
from ._openai_client import OpenAIInterface


HistoryFactory = Callable[[], HistoryInterface]


class AssistantInterface(abc.ABC):
    @abc.abstractmethod
    def exchange(self, user_id: str, question: str) -> str:
        pass


class Assistant(AssistantInterface):
    def __init__(self, openai_client: OpenAIInterface, history_factory: HistoryFactory) -> None:
        self._openai_client = openai_client
        self._history_factory = history_factory
        self._histories: dict[str, HistoryInterface] = {}

    def __get_history(self, user_id: str) -> HistoryInterface:
        if user_id in self._histories:
            return self._histories[user_id]
        self._histories[user_id] = self._history_factory()
        return self._histories[user_id]

    def exchange(self, user_id: str, question: str) -> str:
        history = self.__get_history(user_id)
        history.insert(Role.USER, question)
        response = self._openai_client.exchange(history)
        history.insert(Role.ASSISTANT, response)
        return response
