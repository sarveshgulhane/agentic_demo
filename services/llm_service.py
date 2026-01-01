from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(
    model="ministral-3:3b",
    temperature=0.7,
)


def get_llm_response(prompt: str):
    """LangChain LLM with local ollama models."""

    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content
