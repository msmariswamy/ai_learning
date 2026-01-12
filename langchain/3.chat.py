import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import time
from langchain_openai import ChatOpenAI

from pydantic import SecretStr
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_core.runnables import RunnablePassthrough

start_time = time.time()  # Start timer
load_dotenv(dotenv_path=".env")

# 🔸 Get configuration from environment
llm_provider = os.getenv("LLM_PROVIDER", "groq").lower()

# 🔸 Initialize LLM based on provider

lm_studio_base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
lm_studio_model = os.getenv("LM_STUDIO_MODEL", "local-model")
llm = ChatOpenAI(
    model=lm_studio_model,
    temperature=0,
    openai_api_key="local_api_key",
    openai_api_base=lm_studio_base_url
)
print(f"✅ Using LM Studio (Local) - {lm_studio_base_url}")

# 🔸 Create a Runnable chain
chain_qa = PromptTemplate.from_template("Answer clearly: {question}") | llm
summary_qa = PromptTemplate.from_template("Summarize the following text: {text}") | llm

# New code starts here
# 🔸 Create a Tool for QuestionAnswering - new here
qa_tool = Tool(
    name="QA Tool",
    func=chain_qa.invoke,
    description="A basic LLM tool to answers clearly to asked questions"
)

summary_tool = Tool(
    name="Summary Tool",
    func=summary_qa.invoke,
    description="A basic LLM tool to Summarize the text"
)

# 🚀 Run a query
query = "What is LangChain. Make sure you use latest library?"
response = qa_tool.invoke({"question": query})

text_to_summarize = """LangChain is a framework for developing applications powered by language models. It provides a standard interface for all LLMs, as well as a toolkit to build with them. LangChain enables developers to create applications that can interact with various data sources, manage prompts, and chain together multiple LLM calls to accomplish complex tasks."""
summary_response = summary_tool.invoke({"text": text_to_summarize})

# 🖨️ Output
print("\nUser Question:", query)
print("\nLLM Answer:", response.content if hasattr(response, 'content') else response)
print("\nSummary:", summary_response.content if hasattr(summary_response, 'content') else summary_response)

end_time = time.time()  # End timer
elapsed_time = end_time - start_time  # Calculate duration
print(f"\n⏱️ Time taken: {elapsed_time:.2f} seconds")  # Add this line