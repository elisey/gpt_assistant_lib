class GptAssistantBaseException(Exception):
    """GPT assistant base exception."""


class OpenAICommunicationError(GptAssistantBaseException):
    """Error communication with openai service."""
