# 🍺 Beer Game API v1.1

## 🔐 Authentication
- JWT with HTTP-only cookies
- Required header: `X-CSRF-Token` for non-GET requests
- Login: `POST /auth/login`
- Get current user: `GET /auth/me`

## 🎮 Game Endpoints

### Create Game
```http
POST /games/
```
**Request:**
```json
{
  "name": "Supply Chain Challenge",
  "max_rounds": 20,
  "player_count": 4,
  "demand_pattern": {
    "type": "step",
    "params": {
      "initial_demand": 4,
      "step_round": 5,
      "step_size": 2
    }
  }
}
```

### Join Game
```http
POST /games/{game_id}/join
```
**Request:**
```json
{"role": "retailer"}
```

### Submit Order
```http
POST /games/{game_id}/orders
```
**Request:**
```json
{
  "round": 1,
  "quantity": 8,
  "type": "regular"
}
```

### Get Game State
```http
GET /games/{game_id}/state
```
**Response:**
```json
{
  "game_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "in_progress",
  "current_round": 5,
  "players": [
    {
      "id": 42,
      "role": "retailer",
      "inventory": 15,
      "backlog": 0,
      "last_order": 8
    }
  ]
}
```

## 🔄 WebSocket
```
ws://localhost:8000/ws/game/{game_id}
```

## ⚠️ Error Responses
- `400 Bad Request`: Invalid parameters
- `401 Unauthorized`: Invalid/missing token
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded

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
      "initial_demand": 4,
      "change_week": 6,
      "final_demand": 8
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
  - `initial_demand`: Customer demand before the change (default: 4)
  - `change_week`: Week number at which demand shifts to the new level (default: 6)
  - `final_demand`: Customer demand after the change (default: 8)

Example:
```json
{
  "type": "classic",
  "params": {
    "initial_demand": 4,
    "change_week": 6,
    "final_demand": 8
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
