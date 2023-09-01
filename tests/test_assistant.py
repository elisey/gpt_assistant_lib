from unittest.mock import Mock, call

import pytest

from gpt_assistant_lib._assistant import Assistant, HistoryFactory
from gpt_assistant_lib._history import HistoryInterface, Role
from gpt_assistant_lib._openai_client import OpenAIInterface


@pytest.fixture
def history_factory() -> HistoryFactory:
    return lambda: Mock(spec=HistoryInterface)


@pytest.fixture
def openai_client() -> OpenAIInterface:
    return Mock(spec=OpenAIInterface)


def test_exchange(history_factory, openai_client):
    assistant = Assistant(openai_client, history_factory)

    user_1 = "test_user_1"
    question_1 = "What is the capital of France?"
    answer_1 = "Paris"
    openai_client.exchange.return_value = answer_1
    response = assistant.exchange(user_1, question_1)
    assert response == answer_1

    user_2 = "test_user_2"
    question_2 = "What is the capital of England?"
    answer_2 = "Paris"
    openai_client.exchange.return_value = answer_2
    response = assistant.exchange(user_2, question_2)
    assert response == answer_2

    history_mock: Mock = assistant._histories[user_1]
    history_mock.insert.assert_has_calls([call(Role.USER, question_1), call(Role.ASSISTANT, answer_1)])

    history_mock: Mock = assistant._histories[user_2]
    history_mock.insert.assert_has_calls([call(Role.USER, question_2), call(Role.ASSISTANT, answer_2)])

    assert len(assistant._histories) == 2


if __name__ == "__main__":
    pytest.main()
