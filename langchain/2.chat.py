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

# 🔸 Create Prompt Template
prompt = PromptTemplate.from_template("Answer clearly: {question}")

# 🔸 Create a Runnable chain
chain = prompt | llm

# New code starts here
# 🔸 Create a Tool for QA - new here
qa_tool = Tool(
    name="QA Tool",
    func=chain.invoke,
    description="A basic LLM tool to answers clearly to asked questions"
)

# 🚀 Run a query
query = "What is LangChain. Make sure you use latest library?"
response = qa_tool.invoke(query)

# 🖨️ Output
print("\nUser Question:", query)
print("\nLLM Answer:", response.content if hasattr(response, 'content') else response)

end_time = time.time()  # End timer
elapsed_time = end_time - start_time  # Calculate duration
print(f"\n⏱️ Time taken: {elapsed_time:.2f} seconds")  # Add this line