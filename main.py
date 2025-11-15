from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

class AgentState(TypedDict):
    hero_picks: dict[str, list[str]]
    analysis: str

def expert_agent_node(state: AgentState) -> AgentState:
    """Agent that uses OpenAI Responses API to process messages."""
    hero_picks = state.get("hero_picks", {})
    
    if hero_picks:

        response = client.responses.create(
            model="gpt-5-nano",
            instructions='''
            You are a expert analyst of DOTA 2 professional matches. 
            You will be given the hero picks for each team. 
            Your task is to analyze the picks, provide a detailed analysis of the picks, and predict the outcome of the match coupled with a degree of confidence.
            ''',
            input=f"Analyze the following hero picks: {hero_picks}"
        )
        
        assistant_response = response.output_text
        return {"analysis": assistant_response}
    
    return {"analysis": "No hero picks provided"}


def create_agent_graph():

    graph = StateGraph(AgentState)
    
    graph.add_node("agent", expert_agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    
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

