from ._assistant import Assistant, AssistantInterface
from ._exceptions import GptAssistantBaseException, OpenAICommunicationError
from ._history import HistoryInterface, SimpleHistory
from ._openai_client import OpenAI


__all__ = [
    "AssistantInterface",
    "build_assistant",
    "GptAssistantBaseException",
    "OpenAICommunicationError",
]


def build_assistant(openai_api_key: str, prompt: str, history_size: int, history_ttl: int) -> AssistantInterface:
    """
    Build and configure an assistant instance for interactive conversations.

    This function creates and configures an assistant instance that can engage in interactive
    conversations with users. It combines the OpenAI API client and a conversation history
    manager to create a fully functional assistant.

    Parameters:
    - openai_api_key (str): The OpenAI API key used for communication with the GPT model.
      generate it here - https://platform.openai.com/account/api-keys
    - prompt (str): The initial prompt or system message that sets the context for the conversation.
      more info - https://platform.openai.com/docs/guides/gpt-best-practices/tactic-ask-the-model-to-adopt-a-persona
    - history_size (int): The maximum number of messages to retain in the conversation history.
      Higher number - more context assistent will remember but api calls will cost more.
    - history_ttl (int): The time-to-live (TTL) in seconds for retaining messages in the history.

    Returns:
    - AssistantInterface: An instance of the assistant that can be used for conversation exchanges.

    Example:

    ```python
    openai_api_key = "your_openai_api_key"
    initial_prompt = "You are a helpful assistant."
    max_history_size = 5
    history_lifetime = 600  # 10 minutes
    assistant = build_assistant(openai_api_key, initial_prompt, max_history_size, history_lifetime)
    while True:
        user_input = input("You: ")
        assistant_response = assistant.exchange("any", user_input)
        print(f"Assistant: {assistant_response}")
    ```

    Note:
    - The `AssistantInterface` returned by this function can be used to interact with users
      by exchanging messages using the `exchange` method.
    - The conversation history is managed with a maximum size of `history_size` and a TTL of
      `history_ttl` seconds, allowing for context-aware responses.
    """

    def build_history() -> HistoryInterface:
        history = SimpleHistory(history_size, history_ttl)
        history.init_system_content(prompt)
        return history

    openai_client = OpenAI(openai_api_key)

    return Assistant(openai_client, build_history)
