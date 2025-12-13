import pandas as pd
import json

def get_heroes_stats_tools():

    tools = [
        {
            "type": "function",
            "name": "get_heroes_stats",
            "description": "Get stats for DOTA 2 hero relevent for the current patch (7.39e). Stats include the number of matches the hero was picked, number of wins, and winrate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hero_names": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "The name of the DOTA 2 hero (e.g., 'Abaddon', 'Enigma', 'Slardar')"
                        }
                    }
                },
                "required": ["hero_names"]
            }
        },
    ]
    return tools

def get_abilities_tools():
    tools = [
        {
            "type": "function",
            "name": "get_abilities",
            "description": "Get abilities for DOTA 2 hero in the current patch (7.39e). Abilities include the name of the ability, description, and ability-specific stats.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hero_name": {
                        "type": "string",
                        "description": "The name of the DOTA 2 hero (e.g., 'Abaddon', 'Enigma', 'Slardar')"
                    }
                },
                "required": ["hero_name"]
            }
        },
    ]
    return tools

def get_heroes_stats(hero_names: list[str] ) -> str:
    """Actual implementation of get_heroes_stats tool."""
    df = pd.read_csv("hero_stats.csv")
    heroes_stats = "Hero,Picks,Wins,WinRate\n"
    for hero_name in hero_names:
        hero_data = df[df["hero"].str.lower() == hero_name.lower()]
        if hero_data.empty:
            heroes_stats += f"{hero_name},not found,not found,not found\n"
        else:
            row = hero_data.iloc[0]
            heroes_stats += f"{row['hero']},{row['picks']},{row['wins']},{row['wr%']}%\n"
    return heroes_stats

def get_abilities(hero_name: str) -> dict:
    """Actual implementation of get_abilities tool."""
    with open("abilities.json", "r") as f:
        abilities = json.load(f)
    for index, hero in enumerate(abilities):
        if hero["hero"].lower() == hero_name.lower():
            # Normalize to always return an object
            return abilities[index]
    # Normalize to always return an object (avoid mixing str/dict in graph state)
    return {"hero": hero_name, "error": "Abilities info not found"}