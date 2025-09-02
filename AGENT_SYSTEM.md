# Beer Game Agent System

This document explains how to use the AI agent system in The Beer Game.

## Overview

The agent system allows you to:
1. Create games with AI players
2. Configure different strategies for each agent
3. Toggle demand visibility for agents
4. Simulate full games programmatically

## Agent Strategies

Each agent can use one of these strategies:

- **naive**: Orders exactly what was demanded
- **bullwhip**: Tends to over-order when demand increases
- **conservative**: Maintains stable orders
- **random**: Makes random orders (for testing)

## API Endpoints

### Create a new agent game
```http
POST /api/v1/agent-games/
{
    "name": "AI Game",
    "max_rounds": 20,
    "demand_pattern": {
        "type": "classic",
        "params": {
            "stable_period": 5,
            "step_increase": 4
        }
    }
}
```

### Start the game
```http
POST /api/v1/agent-games/{game_id}/start
```

### Play a round
```http
POST /api/v1/agent-games/{game_id}/play-round
```

### Set agent strategy
```http
PUT /api/v1/agent-games/{game_id}/agent-strategy?role=retailer&strategy=bullwhip
```

### Toggle demand visibility
```http
PUT /api/v1/agent-games/{game_id}/demand-visibility?visible=true
```

### Get game state
```http
GET /api/v1/agent-games/{game_id}/state
```

## Running the Demo

1. Start the backend server:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. In a new terminal, run the demo script:
   ```bash
   cd backend
   python3 -m scripts.demo_agents
   ```

## Customizing the Demo

Edit `backend/scripts/demo_agents.py` to:
- Change agent strategies
- Modify the number of rounds
- Adjust demand patterns
- Toggle demand visibility

## Implementation Details

- Agents are implemented in `backend/app/services/agents.py`
- The game service is in `backend/app/services/agent_game_service.py`
- API endpoints are defined in `backend/app/api/endpoints/agent_game.py`

## Troubleshooting

- Ensure the backend server is running before starting the demo
- Check that your database is properly configured
- Verify that all required Python packages are installed
- Check the backend logs for any error messages
