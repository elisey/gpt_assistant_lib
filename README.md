# GPT-Assistant Library

The GPT-Assistant Library is a Python-based toolkit for building intelligent personal assistants powered by OpenAI's GPT
models. This library simplifies the integration of conversational AI into your applications, making it easier than ever
to create chatbots, virtual assistants, and more.

## Key Features

- Seamless integration with GPT-3 and GPT-4 models.
- Easily manage conversations and user interactions.
- Support multiuser chat history support
- Customizable and extensible to suit your specific needs.
- Developer-friendly API for smooth integration into your projects.

## Installation

```shell
pip install gpt-assistant-lib
```

## Configuration

To set up the assistant, you need an API key and to create a system prompt.

### Obtain API key

Go to Openai [website](https://platform.openai.com/account/api-keys)

### System prompt

It will define your assistant behaviour.
More info you can find
in [documentation](https://platform.openai.com/docs/guides/gpt-best-practices/tactic-ask-the-model-to-adopt-a-persona)

### History

The library maintains a conversation history that the assistant relies on to understand and maintain context during
interactions. This conversation history is included in every request sent to OpenAI. A larger history size allows
the assistant to remember more context, enhancing its ability to engage in meaningful conversations.
However, it's important to note that larger conversation histories can result in increased API call costs.

## Usage

```python
import gpt_assistant_lib

openai_api_key = "your_openai_api_key"
initial_prompt = "You are a helpful assistant."
max_history_size = 5
history_lifetime = 600  # 10 minutes
assistant = gpt_assistant_lib.build_assistant(openai_api_key, initial_prompt, max_history_size, history_lifetime)
while True:
    user_input = input("You: ")
    assistant_response = assistant.exchange("any", user_input)
    print(f"Assistant: {assistant_response}")
```

## Development

> Run with enabled virtual environment

Autoformat source code

```shell
task format
```

Run all linters

```shell
task lint
```

Run unit tests

```shell
task test
```

Run all linters and unit tests

```shell
task all
```

## License

This project is licensed under the MIT License
