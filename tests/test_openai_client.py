from unittest.mock import Mock, patch

import openai
import pytest

from gpt_assistant_lib._exceptions import OpenAICommunicationError
from gpt_assistant_lib._history import HistoryInterface
from gpt_assistant_lib._openai_client import OpenAI


def test_handle_empty_hisotry():
    openai_client = OpenAI("my_key")

    mock_history = Mock(spec=HistoryInterface)
    mock_history.get.return_value = []

    with pytest.raises(AssertionError):
        openai_client.exchange(mock_history)


def test_handle_connectivity_error():
    openai_client = OpenAI("my_key")

    mock_history = Mock(spec=HistoryInterface)
    mock_history.get.return_value = ["foo", "bar"]

    with patch("gpt_assistant_lib._openai_client.openai.ChatCompletion.create", side_effect=openai.OpenAIError):
        with pytest.raises(OpenAICommunicationError):
            openai_client.exchange(mock_history)
