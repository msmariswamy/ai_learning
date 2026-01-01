
from groq import Groq

import os

#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
client = Groq(api_key=os.environ.get("OPENAI_API_KEY"))

messages =[]
system_msg = input ("What type of chatbot would you like to create?\n")
messages.append({"role": "system", "content": system_msg})
print ("Your new assistant is ready!") 

while input != "quit()":
    message = input ("You: ")
    messages.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0)
    
    #here teamprature is crativeness and randomess. 0=no creativeness, 0.8-1=creativeness and random
    reply = (response.choices[0].message.content)
    messages.append({"role": "assistant", "content": reply})

