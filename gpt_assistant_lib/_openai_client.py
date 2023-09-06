import abc

import openai

from ._exceptions import OpenAICommunicationError
from ._history import HistoryInterface


class OpenAIInterface(abc.ABC):
    @abc.abstractmethod
    def exchange(self, history: HistoryInterface) -> str:
        """
        Generate a response using OpenAI's GPT-3.5 Turbo model based on conversation history.

        Parameters:
        - history (HistoryInterface): An object that implements the `HistoryInterface`
          and holds the conversation history, including user and assistant messages.

        Returns:
        - str: The generated response from the GPT-3.5 Turbo model.

        Raises:
        - OpenAICommunicationError: If there is an issue communicating with the OpenAI API
          or if an error is encountered during the response generation.

        Preconditions:
        - The `history` object should contain a conversation with at least one message.
        """
        pass


class OpenAI(OpenAIInterface):
    def __init__(self, openai_api_key: str) -> None:
        openai.api_key = openai_api_key

    def exchange(self, history: HistoryInterface) -> str:
        conversation = history.get()
        assert conversation

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=conversation
            )  # type:ignore[no-untyped-call]
            response: str = completion.choices[0].message.content
            return response
        except openai.OpenAIError as e:
            raise OpenAICommunicationError(str(e))
