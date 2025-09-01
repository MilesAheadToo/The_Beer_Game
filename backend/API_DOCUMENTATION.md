# Beer Game API Documentation

This document provides detailed information about the Beer Game API endpoints, including request/response formats and examples.

## Base URL
All API endpoints are relative to the base URL:
```
http://localhost:8000/api/v1
```

## Authentication
All endpoints require authentication. Include a valid JWT token in the `Authorization` header:
```
Authorization: Bearer <your_jwt_token>
```

## Game Endpoints

### Create a New Game
```
POST /games/
```

**Request Body:**
```json
{
  "name": "My Beer Game",
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

**Response (201 Created):**
```json
{
  "id": 1,
  "name": "My Beer Game",
  "status": "created",
  "current_round": 0,
  "max_rounds": 20,
  "created_at": "2023-04-01T10:00:00Z",
  "updated_at": "2023-04-01T10:00:00Z"
}
```

### Get Game Details
```
GET /games/{game_id}
```

**Response (200 OK):**
```json
{
  "id": 1,
  "name": "My Beer Game",
  "status": "in_progress",
  "current_round": 3,
  "max_rounds": 20,
  "created_at": "2023-04-01T10:00:00Z",
  "updated_at": "2023-04-01T10:05:30Z"
}
```

### Start a Game
```
POST /games/{game_id}/start
```

**Response (200 OK):**
```json
{
  "id": 1,
  "name": "My Beer Game",
  "status": "in_progress",
  "current_round": 1,
  "max_rounds": 20,
  "created_at": "2023-04-01T10:00:00Z",
  "updated_at": "2023-04-01T10:06:15Z"
}
```

### Get Game State
```
GET /games/{game_id}/state
```

**Response (200 OK):**
```json
{
  "id": 1,
  "name": "My Beer Game",
  "status": "in_progress",
  "current_round": 3,
  "max_rounds": 20,
  "demand_pattern": {
    "type": "classic",
    "params": {
      "stable_period": 5,
      "step_increase": 4
    },
    "current_demand": 8,
    "next_demand": 8
  },
  "players": [
    {
      "id": 1,
      "name": "Retailer 1",
      "role": "retailer",
      "is_ai": false,
      "current_stock": 10,
      "incoming_shipments": [{"quantity": 4, "arrival_round": 5}],
      "backorders": 0,
      "total_cost": 15.50
    }
  ]
}
```

## Player Endpoints

### Add Player to Game
```
POST /games/{game_id}/players
```

**Request Body:**
```json
{
  "name": "Retailer 1",
  "role": "retailer",
  "is_ai": false
}
```

**Response (201 Created):**
```json
{
  "id": 1,
  "game_id": 1,
  "name": "Retailer 1",
  "role": "retailer",
  "is_ai": false,
  "user_id": 123
}
```

### List Players in Game
```
GET /games/{game_id}/players
```

**Response (200 OK):**
```json
[
  {
    "id": 1,
    "game_id": 1,
    "name": "Retailer 1",
    "role": "retailer",
    "is_ai": false,
    "user_id": 123
  },
  {
    "id": 2,
    "game_id": 1,
    "name": "Wholesaler 1",
    "role": "wholesaler",
    "is_ai": true,
    "user_id": null
  }
]
```

## Order Endpoints

### Submit Order
```
POST /games/{game_id}/players/{player_id}/orders
```

**Request Body:**
```json
{
  "quantity": 5
}
```

**Response (201 Created):**
```json
{
  "id": 1,
  "game_id": 1,
  "player_id": 1,
  "round_number": 3,
  "quantity": 5,
  "created_at": "2023-04-01T10:15:30Z"
}
```

## Round Endpoints

### Get Current Round
```
GET /games/{game_id}/rounds/current
```

**Response (200 OK):**
```json
{
  "id": 3,
  "game_id": 1,
  "round_number": 3,
  "customer_demand": 8,
  "created_at": "2023-04-01T10:10:00Z",
  "player_rounds": [
    {
      "id": 5,
      "player_id": 1,
      "round_id": 3,
      "order_placed": 5,
      "order_received": 4,
      "inventory_before": 6,
      "inventory_after": 2,
      "backorders_before": 0,
      "backorders_after": 0,
      "holding_cost": 1.0,
      "backorder_cost": 0.0,
      "total_cost": 1.0
    }
  ]
}
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid request data"
}
```

### 401 Unauthorized
```json
{
  "detail": "Not authenticated"
}
```

### 403 Forbidden
```json
{
  "detail": "Not enough permissions"
}
```

### 404 Not Found
```json
{
  "detail": "Game not found"
}
```

## Demand Pattern Types

The game supports different demand patterns that can be specified when creating a game:

### Classic Pattern
- **Type:** `classic`
- **Parameters:**
  - `stable_period`: Number of initial rounds with stable demand (default: 5)
  - `step_increase`: Increase in demand after stable period (default: 4)

Example:
```json
{
  "type": "classic",
  "params": {
    "stable_period": 5,
    "step_increase": 4
  }
}
```

### Random Pattern
- **Type:** `random`
- **Parameters:**
  - `min_demand`: Minimum possible demand (default: 2)
  - `max_demand`: Maximum possible demand (default: 12)

### Seasonal Pattern
- **Type:** `seasonal`
- **Parameters:**
  - `base_demand`: Base demand level (default: 8)
  - `amplitude`: Amplitude of seasonal variation (default: 4)
  - `period`: Number of rounds in a full cycle (default: 12)
