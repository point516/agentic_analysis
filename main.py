from openai.types.responses import ResponseFunctionToolCall
from tools import get_heroes_stats_tools, get_heroes_stats, get_abilities_tools, get_abilities

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.types import RetryPolicy
from openai import OpenAI
from openai.types.responses import ResponseFunctionToolCall
from dotenv import load_dotenv
import os
import json

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------- ERRORS --------------------

class NoToolCallsError(Exception):
    pass

# ------------------- STATE --------------------

class AgentState(TypedDict): # it is better to regard state as read-only
    hero_picks: dict[str, list[str]]
    analysis: str | None
    heroes_stats: str | None

    abilities_by_hero: dict[str, dict] | None
    # Safety counter to prevent infinite tool loops.
    abilities_steps: int | None
    tool_calls: list[ResponseFunctionToolCall] | None # also can use Annotated and specify the operator from Langgraph (for different logic of state updates)

# ------------------- TOOL CALL ROUTING --------------------

def tool_call_hs(state: AgentState) -> Literal["get_heroes_stats", "abilities_agent"]: # can also returb a list of nodes, they will be executed in parallel
    if state.get("tool_calls") and state.get("tool_calls")[-1].name == "get_heroes_stats":
        print(f"Directing to get_heroes_stats tool execution node")
        return "get_heroes_stats" 
    print(f"No more get_heroes_stats tool calls needed, Directing to abilities_agent node")
    return "abilities_agent"

def tool_call_abilities(state: AgentState) -> Literal["get_abilities", "expert_agent"]:
    # Safety stop to avoid infinite loops.
    if (state.get("abilities_steps") or 0) >= 12:
        return "expert_agent"

    tool_calls = state.get("tool_calls") or []
    if tool_calls and tool_calls[-1].name == "get_abilities":
        print(f"Directing to get_abilities tool execution node")
        return "get_abilities"
    
    print(f"No more get_abilities tool calls needed, Directing to expert_agent node")
    return "expert_agent"

# ------------------- TOOL CALL EXECUTION -------------------

def get_heroes_stats_node(state: AgentState) -> AgentState:
    tool_calls = state.get("tool_calls") or []
    if not tool_calls:
        return {}
    
    executed_tool_call = tool_calls[-1]
    tool_call_id = executed_tool_call.id
    
    arguments = json.loads(executed_tool_call.arguments)
    heroes = arguments["hero_names"]
    heroes_stats = get_heroes_stats(heroes)
    
    remaining_tool_calls = [tc for tc in tool_calls if tc.id != tool_call_id]
    
    return {
        "heroes_stats": heroes_stats,
        "tool_calls": remaining_tool_calls if remaining_tool_calls else None
    }

def get_abilities_node(state: AgentState) -> AgentState:
    tool_calls = state.get("tool_calls") or []
    if not tool_calls:
        return {}
    
    executed_tool_call = tool_calls[-1]
    tool_call_id = executed_tool_call.id
    
    arguments = json.loads(executed_tool_call.arguments)
    hero_name = arguments["hero_name"]
    abilities = get_abilities(hero_name)
    hero_key = abilities.get("hero", hero_name)
    
    remaining_tool_calls = [tc for tc in tool_calls if tc.id != tool_call_id]
    
    return {
        "abilities_by_hero": {
            **(state.get("abilities_by_hero") or {}), # unpacking the dict and adding the new ability
            hero_key: abilities,
        },
        "abilities_steps": (state.get("abilities_steps") or 0) + 1,
        "tool_calls": remaining_tool_calls if remaining_tool_calls else None
    }

# ------------------- AGENTS --------------------

def heroes_stats_agent_node(state: AgentState) -> AgentState:

    hero_picks = state.get("hero_picks")

    response = client.responses.create(
        model="gpt-5-mini",
        instructions='''
            You are a helper agent that will help the main agent with the analysis of the DOTA 2 hero picks. 
            Your task is to provide the main agent with the current patch stats of the heroes that were picked.
            For that you have a 'get_heroes_stats' tool. 
            ''',
        input=f"Provide the current patch stats of the heroes that were picked: {json.dumps(hero_picks)}",
        reasoning={"effort":"low"},
        tools=get_heroes_stats_tools()
    )
    new_tool_calls: list[ResponseFunctionToolCall] = []

    for item in response.output:
        if item.type == "function_call" and item.name == "get_heroes_stats":
            new_tool_calls.append(item)
            print(f"New tool call issued to get heroes stats for heroes: {json.loads(item.arguments).get('hero_names')}")

    if not new_tool_calls:
        print(f"No tool calls found for heroes stats, raising NoToolCallsError")
        raise NoToolCallsError("No tool calls found") # TODO: add retry policy

    return { # can also return Command and route to tool call if there is a tool call
        "tool_calls": (state.get("tool_calls") or []) + new_tool_calls
    }

def get_abilities_agent_node(state: AgentState) -> AgentState:
    hero_picks = state.get("hero_picks")
    abilities_by_hero = state.get("abilities_by_hero") or {}
    already_fetched = sorted(list(abilities_by_hero.keys()))

    print(f"Currently fetched abilities for heroes: {already_fetched}")

    response = client.responses.create(
        model="gpt-5-mini",
        instructions='''
            You are a helper agent that will help the main agent with the analysis of the DOTA 2 hero picks. 
            Your task is to provide the main agent with the current information about the abilities of the heroes that were picked.
            Current patch is 7.39e. 
            So, if you think that you need to update or check the correctness of your knowledge, 
            for that you have a 'get_abilities' tool. 

            IMPORTANT:
            - Call 'get_abilities' for EXACTLY ONE hero at a time.
            - Only call the tool for heroes you still need info for (do NOT repeat heroes already fetched).
            - If you have enough info and no more tool calls are needed, do not call any tools.
            ''',
        input=f"""
            Hero picks: {json.dumps(hero_picks)}
            Abilities already fetched for these heroes: {json.dumps(already_fetched)}
            If you need information about current abilities for any other hero, call the tool.
            Otherwise, do not call any tools.
        """,
        reasoning={"effort":"low"},
        tools=get_abilities_tools()
    )

    new_tool_calls: list[ResponseFunctionToolCall] = []

    for item in response.output:
        if item.type == "function_call" and item.name == "get_abilities":
            new_tool_calls.append(item)
            print(f"New tool call issued to get abilities for hero: {json.loads(item.arguments).get('hero_name')}")

    return {
        "tool_calls": (state.get("tool_calls") or []) + new_tool_calls
    }

def expert_agent_node(state: AgentState) -> AgentState:
    """Agent that uses OpenAI Responses API to process messages."""
    hero_picks = state.get("hero_picks", {})
    heroes_stats = state.get("heroes_stats", "")
    abilities_by_hero = state.get("abilities_by_hero") or {}
    
    if hero_picks:

        response = client.responses.create(
            model="gpt-5-mini",
            instructions='''
            You are a expert analyst of DOTA 2 professional matches. 
            You will be given the hero picks for each team and additional information, stats of the heroes that were picked. 
            Your task is to analyze the picks, provide analysis, and predict the outcome of the match coupled with a degree of confidence.
            ''',
            input=f'''
            Analyze the following hero picks: {hero_picks}
            Stats in the current patch: {heroes_stats}
            Abilities information fetched: {json.dumps(abilities_by_hero)}
            ''',
            reasoning={"effort":"low"},
        )
        
        assistant_response = response.output_text
        return {"analysis": assistant_response}
    
    return {"analysis": "No hero picks provided"}

# ------------------- GRAPH --------------------

def create_agent_graph():

    graph = StateGraph(AgentState)

    # nodes for tool calls
    graph.add_node("get_heroes_stats", get_heroes_stats_node)
    graph.add_node("get_abilities", get_abilities_node)

    # agents
    graph.add_node("heroes_stats_agent", heroes_stats_agent_node, retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0))
    graph.add_node("abilities_agent", get_abilities_agent_node)
    graph.add_node("expert_agent", expert_agent_node)

    # flow
    graph.add_edge(START, "heroes_stats_agent")
    graph.add_conditional_edges("heroes_stats_agent", tool_call_hs) # as a third parameter can pass a dict to map the output to nodes
    graph.add_edge("get_heroes_stats", "abilities_agent")
    graph.add_conditional_edges("abilities_agent", tool_call_abilities)

    # ReAct loop: after executing the tool, think again (unless we're done / hit safety stop).
    graph.add_edge("get_abilities", "abilities_agent")
    graph.add_edge("expert_agent", END)

    # if from one node, more than 1 nodes, they will be executed in parallel, just be careful about the state updates https://docs.langchain.com/oss/python/langgraph/workflows-agents#parallelization
    return graph.compile()

# ------------------- MAIN --------------------

if __name__ == "__main__":
    agent_graph = create_agent_graph()
    
    initial_state = {
        "hero_picks": {
            "radiant": ["Abaddon", "Enigma", "Slardar", "Hoodwink", "Shadow Demon"],
            "dire": ["Templar Assasin", "Queen of Pain", "Centaur Warrunner", "Enchantress", "Naga Siren"]
        }
    }
    
    result = agent_graph.invoke(initial_state)
    with open("output.json", "w") as f:
        json.dump(result, f, indent=2)