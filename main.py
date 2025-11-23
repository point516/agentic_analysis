from tools import get_heroes_stats_tools, get_heroes_stats

from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import os
import json

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

class AgentState(TypedDict): # it is better to regard state as read-only
    hero_picks: dict[str, list[str]]
    analysis: str

    tool_calls: list[dict] # also can use Annotated and specify the operator from Langgraph (for different logic of state updates)

def get_hero_stats_node(state: AgentState) -> AgentState:
    pass

def heroes_stats_agent_node(state: AgentState) -> AgentState:

    hero_picks = state.get("hero_picks", {})

    response = client.responses.create(
        model="gpt-5-mini",
        instructions='''
            You are a helper agent that will help the main agent with the analysis of the DOAT 2 hero picks. 
            Your task is to provide the main agent with the current patch stats of the heroes that were picked.
            For that you have a 'get_heroes_stats' tool. 
            ''',
        input=f"Provide the current patch stats of the heroes that were picked: {json.dumps(hero_picks)}",
        reasoning={"effort":"low"},
        tools=get_heroes_stats_tools()
    )
    new_tool_calls: list[dict] = []

    for item in response.output:
        if item.type == "function_call" and item.name == "get_heroes_stats":
            new_tool_calls.append(json.loads(item.arguments))

    if not new_tool_calls:
        raise ValueError("No tool calls found") # TODO: add retry policy

    return {
        "tool_calls": state.get("tool_calls", []) + new_tool_calls
    }

def expert_agent_node(state: AgentState) -> AgentState:
    """Agent that uses OpenAI Responses API to process messages."""
    hero_picks = state.get("hero_picks", {})
    
    if hero_picks:

        response = client.responses.create(
            model="gpt-5-mini",
            instructions='''
            You are a expert analyst of DOTA 2 professional matches. 
            You will be given the hero picks for each team and additional information, stats of the heroes that were picked. 
            Your task is to analyze the picks, provide analysis, and predict the outcome of the match coupled with a degree of confidence.
            ''',
            input=f"Analyze the following hero picks: {hero_picks}",
            reasoning={"effort":"low"},
        )
        
        assistant_response = response.output_text
        return {"analysis": assistant_response}
    
    return {"analysis": "No hero picks provided"}


def create_agent_graph():

    graph = StateGraph(AgentState)
    
    # need to complete this (also retry policy)
    graph.add_node("heroes_stats", heroes_stats_agent_node)
    graph.add_node("expert", expert_agent_node)
    graph.add_edge(START, "heroes_stats")
    graph.add_edge("heroes_stats", "expert")
    graph.add_edge("expert", END)
    
    return graph.compile()


if __name__ == "__main__":
    agent_graph = create_agent_graph()
    
    initial_state = {
        "hero_picks": {
            "radiant": ["Abaddon", "Enigma", "Slardar", "Hoodwink", "Shadow Demon"],
            "dire": ["Templar Assasin", "Queen of Pain", "Centaur Warrunner", "Enchantress", "Naga Siren"]
        }
    }
    
    result = agent_graph.invoke(initial_state)
    print(result)

