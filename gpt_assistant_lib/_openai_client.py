import abc

import openai

from ._history import HistoryInterface


class OpenAIInterface(abc.ABC):
    @abc.abstractmethod
    def exchange(self, history: HistoryInterface) -> str:
        pass


class OpenAI(OpenAIInterface):
    def __init__(self, openai_api_key: str) -> None:
        openai.api_key = openai_api_key

    def exchange(self, history: HistoryInterface) -> str:
        conversation: list[dict[str, str]] = history.get()
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=conversation)
        response: str = completion.choices[0].message.content
        return response
