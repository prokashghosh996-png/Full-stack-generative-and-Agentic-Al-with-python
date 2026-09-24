import json
import os
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from groq import Groq, GroqError


SYSTEM_PROMPTS = """You are a helpful weather assistant.
For current weather, call the provided get_weather function.
If the location is missing, ask for it. Only use the provided tool.
Use the tool result to give a short, friendly answer in plain text.
If the tool reports an error, explain that the lookup failed.
Never invent weather data.
"""

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city or location.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Location, e.g. Goa, India"}
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}]


def get_weather(city: str):
    if not isinstance(city, str) or not city.strip():
        return "Error: Please provide a valid city name."
    city = city.strip()
    try:
        response = requests.get(
            f"https://wttr.in/{quote(city, safe='')}",
            params={"format": "%C %t"},
            timeout=20,
        )
        response.raise_for_status()
        return f"The weather in {city} is {response.text.strip()}"
    except requests.RequestException:
        return f"Error: Could not retrieve weather for {city}."


available_tools = {"get_weather": get_weather}


def run_agent(client, user_query):
    message_history = [
        {"role": "system", "content": SYSTEM_PROMPTS},
        {"role": "user", "content": user_query},
    ]
    print(f"\n🔥 START: {user_query}")
    # PLAN lines are application progress updates.
    print("🧠 PLAN: Check the request and identify the location.")

    for _ in range(6):
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=message_history,
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return message.content or "No answer returned. Please try again."

        message_history.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [call.model_dump() for call in message.tool_calls],
        })
        for call in message.tool_calls:
            tool_name = call.function.name
            function = available_tools.get(tool_name)
            if function is None:
                tool_output = f"Error: Unknown tool '{tool_name}'."
            else:
                try:
                    arguments = json.loads(call.function.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object.")
                    print("🧠 PLAN: Retrieve weather for the requested location.")
                    print(f"🛠️ TOOL: {tool_name}({arguments})")
                    tool_output = function(**arguments)
                except (ValueError, TypeError) as exc:
                    tool_output = f"Error: Invalid tool arguments: {exc}"
            print(f"👀 OBSERVE: {tool_output}")
            message_history.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": tool_output,
            })
        print("🧠 PLAN: Use the observation to prepare the answer.")

    return "Could not finish within the request limit. Please try again."


def main():
    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Set GROQ_API_KEY in weather_agent/.env")
        return
    user_query = input("\n👻 Ask me about the weather: ").strip()
    if not user_query:
        print("🤖 Please enter a question.")
        return
    try:
        with Groq(api_key=api_key) as client:
            answer = run_agent(client, user_query)
            print(f"\n🤖 OUTPUT: {answer}")
    except GroqError as exc:
        print(f"\n❌ API ERROR: {exc}")


if __name__ == "__main__":
    main()
