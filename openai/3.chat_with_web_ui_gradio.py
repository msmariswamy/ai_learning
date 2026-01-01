
from groq import Groq
import gradio

import os

#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
client = Groq(api_key=os.environ.get("OPENAI_API_KEY"))

messages =[]
system_msg = "You are a financial advisor chatbot that specializes in helping users with budgeting, saving, and investing. Provide clear and concise advice while maintaining a friendly and approachable tone."
messages.append({"role": "system", "content": system_msg})
print ("Your new assistant is ready!") 

def chatbot_interface(user_input):
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages)
    reply = (response.choices[0].message.content)
    messages.append({"role": "assistant", "content": reply})
    print("\n" + (reply or "") + "\n")

    return reply

iface = gradio.Interface(
    fn=chatbot_interface,
    inputs="text",
    outputs="text",
    title="Financial Advisor Chatbot",
    description="A chatbot that provides financial advice on budgeting, saving, and investing.",
)
iface.launch()