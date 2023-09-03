import abc
from typing import Callable

from ._exceptions import OpenAICommunicationError
from ._history import HistoryInterface, Role
from ._openai_client import OpenAIInterface


HistoryFactory = Callable[[], HistoryInterface]


class AssistantInterface(abc.ABC):
    @abc.abstractmethod
    def exchange(self, user_id: str, question: str) -> str:
        """
        Perform an interactive exchange of conversation between a user and an assistant.

        Parameters:
        - user_id (str): A unique identifier for the user initiating the conversation.
        - question (str): The user's question or input to be included in the conversation.

        Returns:
        - str: The assistant's response to the user's question.

        Raises:
        - OpenAICommunicationError: If there is an issue communicating with the OpenAI API
          or if an error is encountered during the response generation.
        """

    pass


class Assistant(AssistantInterface):
    def __init__(self, openai_client: OpenAIInterface, history_factory: HistoryFactory) -> None:
        self._openai_client = openai_client
        self._history_factory = history_factory
        self._histories: dict[str, HistoryInterface] = {}

    def _get_history(self, user_id: str) -> HistoryInterface:
        if user_id not in self._histories:
            self._histories[user_id] = self._history_factory()
        return self._histories[user_id]

    def exchange(self, user_id: str, question: str) -> str:
        history = self._get_history(user_id)
        history.insert(Role.USER, question)
        try:
            response = self._openai_client.exchange(history)
        except OpenAICommunicationError:
            raise
        history.insert(Role.ASSISTANT, response)
        return response
