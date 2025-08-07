from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from typing import Annotated
from langgraph.graph.message import add_messages
from IPython.display import display, Image
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
import os
from dotenv import load_dotenv
load_dotenv()


os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

## Langsmith Tracking and tracing

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

model = ChatOpenAI(temperature=0)

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def make_default_graph():
    """Make simple LLM agent graph"""
    graph_workflow = StateGraph(
        State
    )

    def call_model(state):
        return {"messages": [model.invoke(state["messages"])]}
    
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge("agent", END)
    graph_workflow.add_edge(START, "agent")

    agent = graph_workflow.compile()

    return agent

def make_alternative_graph():
    """Make tool-calling agent"""

    @tool
    def add(a: int, b: int) -> int:
        """
        Add two integers.

        Args:
            a (int): The first integer.
            b (int): The second integer.

        Returns:
            int: The sum of a and b.
        """
        return a + b
    
    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools([add])

    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}
    
    def should_continue(state):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END
    
    graph_workflow = StateGraph(
        State
    )
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges(
        "agent", should_continue
    )

    agent = graph_workflow.compile()

    return agent

agent = make_alternative_graph()