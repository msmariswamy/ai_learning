import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

from pydantic import SecretStr
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv(dotenv_path=".env")

# 🔸 Get configuration from environment
llm_provider = os.getenv("LLM_PROVIDER", "groq").lower()

# 🔸 Initialize LLM based on provider
if llm_provider == "groq":
    groq_api_key = SecretStr(os.getenv("GROQ_API_KEY"))
    llm_model = os.getenv("LLM_MODEL", "nvidia/nemotron-3-nano")
    llm = ChatGroq(model=llm_model, temperature=0, api_key=groq_api_key)
    print(f"✅ Using Groq LLM")
elif llm_provider == "lmstudio":
    lm_studio_base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    lm_studio_model = os.getenv("LM_STUDIO_MODEL", "local-model")
    llm = ChatOpenAI(
        model=lm_studio_model,
        temperature=0,
        openai_api_key="local_api_key",
        openai_api_base=lm_studio_base_url
    )
    print(f"✅ Using LM Studio (Local) - {lm_studio_base_url}")
else:
    raise ValueError(f"Invalid LLM_PROVIDER: {llm_provider}. Must be 'groq' or 'lmstudio'")

# 🔸 Create Prompt Template
prompt = PromptTemplate.from_template("Answer clearly: {question}")

# 🔸 Create a Runnable chain
chain = prompt | llm

# 🚀 Run a query
query = "What is LangChain. Make sure you use latest library?"
response = chain.invoke({"question": query})

# 🖨️ Output
print("\nUser Question:", query)
print("\nLLM Answer:", response.content if hasattr(response, 'content') else response)
