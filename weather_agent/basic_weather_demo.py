import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).with_name(".env"))

def get_weather(city:str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)
    if response.status_code ==200:
        return f"The weather in {city} is {response.text}"
    return "something went wrong"
 
def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY in weather_agent/.env before running.")
    user_query =input(">")
    with Groq(api_key=api_key) as client:
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL")or "openai/gpt-oss-20b",
            messages=[{"role": "user", "content": user_query}],
        )
    print(response.choices[0].message.content)

#if __name__ == "__main__":
    #main()

print(get_weather("Assam"))
