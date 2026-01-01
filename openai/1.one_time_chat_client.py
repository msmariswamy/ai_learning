
from groq import Groq

import os

#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
client = Groq(api_key=os.environ.get("OPENAI_API_KEY"))

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "developer", "content": "Talk like a 10 years old Software Architect."},
        {
            "role": "user",
            "content": "Give me 3 ideas for apps I could build with openai apis?",
        },
    ],
)
print("-----\nResponse:\n-----")

print(completion)
print("-----\nResponse:\n-----")
print(completion.choices[0].message.content)