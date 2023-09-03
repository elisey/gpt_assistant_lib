import sys

import gpt_assistant_lib


def main() -> None:
    openai_api_key = "your_openai_api_key"
    initial_prompt = "You are a helpful assistant."
    max_history_size = 5
    history_lifetime = 600  # 10 minutes
    assistant = gpt_assistant_lib.build_assistant(openai_api_key, initial_prompt, max_history_size, history_lifetime)
    while True:
        user_input = input("You: ")
        try:
            assistant_response = assistant.exchange("any", user_input)
        except gpt_assistant_lib.OpenAICommunicationError as e:
            print(f"Something went wrong. {e}")
            sys.exit(1)
        else:
            print(f"Assistant: {assistant_response}")


if __name__ == "__main__":
    main()
