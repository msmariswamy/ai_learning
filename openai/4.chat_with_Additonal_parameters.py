
from groq import Groq

import os

#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
client = Groq(api_key=os.environ.get("OPENAI_API_KEY"))

def get_completion(promot, model = "openai/gpt-oss-120b", max_token=190):
    messages1 = [{'role': 'user', 'content': promot}]
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages1,
        temperature=0,
        max_tokens=max_token
    )
    #print(completion)
    return completion.choices[0].message.content

def get_completion_with_messages(messages1, model = "openai/gpt-oss-120b", max_token=500, temperature=0):
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages1,
        temperature=temperature,
        max_tokens=max_token
    )
    #print(completion)
    return completion.choices[0].message.content

#Assignment 1
#response = get_completion("Take the letters in lollipop and reverse them")
#print(response)

#Assigment 2
messages = [{'role':'system', 'content':'You are an assistant who responds in the style of Dr Seuss. The sentence should be only one line max 2000 character'},{'role': 'user','content':'write me a very short poem about a happy carrot'}]
response = get_completion_with_messages(messages,temperature=1,max_token=1000, model="groq/compound")
print(response)
