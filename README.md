# Experimenting with LangGraph Agents -- Agentic System to analyze DOTA2 hero picks.

## System Architecture

The system uses a multi-agent workflow built with LangGraph:

1. **Heroes Stats Agent**: Fetches current patch statistics (Total matches, Wins, Losses, Win-rate%) for picked heroes. Equipped with `get_heroes_stats` tool.
2. **Abilities Agent**: ReAct Agent (think -> act -> observe) that fetches up-to-date information (for the current patch) about heroes. Equipped with `get_abilities` tool.
3. **Expert Agent**: Analyzes all collected data (hero picks, stats, and abilities) to provide match analysis and predictions

The workflow routes conditionally based on tool calls, allowing agents to decide when to fetch additional data or proceed to analysis. State is managed through a shared `AgentState` that tracks hero picks, stats, abilities, and pending tool calls.
