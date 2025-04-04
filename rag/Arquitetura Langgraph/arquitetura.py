import os
from dotenv import load_dotenv
from langchain_core.messages.tool import tool_call

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import Literal

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.types import Command

model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
@tool
def transfer_to_travel_advisor():
    """Ask travel advisor for help."""
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to another agent
    return


@tool
def transfer_to_hotel_advisor():
    """Ask hotel advisor for help."""
    return

def travel_advisor(state: MessagesState) -> Command[Literal["hotel_advisor", "__end__"]]:
    system_prompt = (
        "Você é um especialista em viagens com foco em destinos brasileiros, capaz de recomendar locais para visitar (ex: estados, cidades, regiões, etc)."
        "Se precisar de recomendações de hotéis, peça ajuda ao 'hotel_advisor'."
    )
    
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = model.bind_tools([transfer_to_travel_advisor]).invoke(messages)
    
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        tool_message = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id
        }
        return Command(goto="hotel_advisor", update={"messages": [ai_msg, tool_message]})
    
    return {"messages": [ai_msg]}
    
def hotel_advisor(state: MessagesState) -> Command[Literal["travel_advisor", "__end__"]]:
    system_prompt = (
        "Você é um especialista em hotéis capaz de fornecer recomendações de hospedagem para um determinado destino. "
        "Se precisar de ajuda para escolher destinos de viagem, peça ajuda ao 'travel_advisor'."
    )
    
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = model.bind_tools([transfer_to_hotel_advisor]).invoke(messages)
    
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        return Command(goto="hotel_advisor", update={"messages": [ai_msg, tool_msg]})

    
    return {"messages": [ai_msg]}
    
    

if __name__ == "__main__":
    builder = StateGraph(MessagesState)
    builder.add_node("travel_advisor", travel_advisor)
    builder.add_node("hotel_advisor", hotel_advisor)
    builder.add_edge(START, "travel_advisor")

    graph = builder.compile()
    
    # initial_input = {"messages": [HumanMessage(content="Eu gostaria de visitar cidades historicas com alta gastronomia local. Escolha um destino e forneca algumas opcoes de hoteis. Gosto muito de comida mineira")]}  
    initial_input = {
        "messages": [
            (
                "user",
                "Eu gostaria de visitar cidades historicas com alta gastronomia local. Escolha um destino e forneca algumas opcoes de hoteis. Gosto muito de comida mineira",
            )
        ]
    }

    config = {"configurable": {"thread_id": "1"}}
    for chunk in graph.stream(initial_input, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()