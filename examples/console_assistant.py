import gpt_assistant_lib


OPENAI_API_KEY = "your_openai_api_key"
PROMPT = "You are helpful assistant"
HISTORY_SIZE = 5
TTL_S = 10 * 60


def main() -> None:
    assistant = gpt_assistant_lib.build_assistant(OPENAI_API_KEY, PROMPT, HISTORY_SIZE, TTL_S)
    while True:
        user_input = input("You: ")
        assistant_response = assistant.exchange("any", user_input)
        print(f"Assistant: {assistant_response}")


if __name__ == "__main__":
    main()
