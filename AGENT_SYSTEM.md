# Beer Game Agent System

This document provides comprehensive documentation for the AI agent system in The Beer Game, including configuration, strategies, and API usage.

## 🎯 Overview

The agent system enables automated gameplay with configurable AI agents that can:
- Participate in games alongside human players
- Use various decision-making strategies
- Adapt to different supply chain scenarios
- Provide insights into supply chain dynamics

## 🧠 Agent Strategies

### Available Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **naive** | Orders exactly the incoming demand | Baseline testing |
| **bullwhip** | Over-orders when demand increases | Demonstrating supply chain volatility |
| **conservative** | Maintains stable order quantities | Minimizing inventory costs |
| **ml_forecast** | Uses machine learning for demand prediction | Realistic demand planning |
| **optimizer** | Optimizes orders based on cost functions | Cost optimization |
| **reactive** | Reacts quickly to inventory changes | Volatile markets |

### Strategy Configuration

Each agent can be configured with strategy-specific parameters:

```json
{
  "strategy": "ml_forecast",
  "params": {
    "lookback_period": 5,
    "safety_stock": 2.0,
    "forecast_horizon": 3
  }
}
```

## 🔌 API Endpoints

### Base URL
All endpoints are relative to: `http://localhost:8000/api/v1`

### Authentication
Include JWT token in the `Authorization` header:
```
Authorization: Bearer <your_jwt_token>
```

### 1. Create Agent Game
```http
POST /agent-games/
```
**Request Body:**
```json
{
  "name": "AI Simulation",
  "max_rounds": 20,
  "player_count": 4,
  "demand_pattern": {
    "type": "step",
    "params": {
      "initial_demand": 4,
      "step_round": 5,
      "step_size": 2
    }
  },
  "agent_configs": [
    {
      "node_id": "retailer",
      "strategy": "ml_forecast",
      "params": {"lookback_period": 5}
    },
    {
      "node_id": "wholesaler",
      "strategy": "conservative",
      "params": {"safety_factor": 1.5}
    }
  ]
}
```

### 2. Start Game
```http
POST /agent-games/{game_id}/start
```
**Response:**
```json
{
  "status": "started",
  "current_round": 1,
  "game_state": { ... }
}
```

### 3. Play Round
```http
POST /agent-games/{game_id}/play-round
```
**Response:**
```json
{
  "round_completed": 2,
  "game_state": { ... },
  "metrics": {
    "inventory_costs": 120.50,
    "backlog_costs": 45.00,
    "service_level": 0.95
  }
}
```

### 4. Get Game State
```http
GET /agent-games/{game_id}
```

### 5. Update Agent Strategy
```http
PATCH /agent-games/{game_id}/agents/{agent_id}
```
**Request Body:**
```json
{
  "strategy": "bullwhip",
  "params": {"aggressiveness": 1.8}
}
```

## 🔄 WebSocket API

Connect to `ws://localhost:8000/ws/game/{game_id}` for real-time updates:

```javascript
const socket = new WebSocket('ws://localhost:8000/ws/game/123');

socket.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log('Game update:', update);
};
```

## 📊 Monitoring and Analytics

### Available Metrics
- Inventory levels
- Order history
- Backlog amounts
- Costs (holding, backlog, total)
- Service level
- Bullwhip effect metrics

### Exporting Data
```http
GET /agent-games/{game_id}/export
```

## 🔧 Configuration

### Environment Variables
```env
# Agent System
AGENT_STRATEGY_DEFAULT=ml_forecast
AGENT_UPDATE_INTERVAL=1000  # ms
MAX_CONCURRENT_AGENTS=10
```

### Strategy Parameters
Each strategy supports different parameters:

**ML Forecast**
- `lookback_period`: Number of previous rounds to consider
- `safety_stock`: Multiplier for safety stock calculation
- `forecast_horizon`: Number of rounds to forecast

**Bullwhip**
- `aggressiveness`: How much to over-order (1.0-3.0)
- `volatility_threshold`: Demand change that triggers over-ordering

**Conservative**
- `safety_factor`: Base safety stock multiplier
- `max_order_change`: Maximum change in order quantity per round
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
