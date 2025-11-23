import pandas as pd

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

def get_heroes_stats(hero_names: list[str]) -> str:
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
