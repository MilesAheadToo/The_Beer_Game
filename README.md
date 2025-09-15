# The Beer Game with Supply Chain Analytics

A comprehensive simulation of the Beer Distribution Game featuring AI-powered supply chain analysis and optimization. This implementation includes:

- **Secure Authentication**: JWT-based authentication with role-based access control
- **Advanced Analytics**: Real-time supply chain metrics and visualization
- **AI-Powered Insights**: Machine learning models for demand forecasting and optimization
- **Multiplayer Support**: Play with both human and AI players
- **Automatic LLM Players**: Default retailer, wholesaler, distributor, and manufacturer players are created as LLM agents when a group is created
- **Containerized Deployment**: Easy setup with Docker and Docker Compose

## 🚀 Features

- **Secure Authentication**
  - JWT-based authentication with HTTP-only cookies
  - CSRF protection with double-submit cookie pattern
  - Role-based access control (Admin, Manager, Player)
  - Secure password hashing with bcrypt

- **Supply Chain Simulation**
  - Configurable supply chain networks
  - Real-time inventory and order tracking
  - Demand forecasting with machine learning
  - Bullwhip effect visualization

- **Admin Dashboard**
  - User management
  - Game configuration
  - System monitoring
  - Analytics and reporting

- **API-First Design**
  - RESTful API with OpenAPI documentation
  - WebSocket support for real-time updates
  - Comprehensive error handling and logging

## 🛠 Tech Stack

- **Frontend**: React 18, Material-UI 5, Redux Toolkit, React Query, Recharts
- **Backend**: FastAPI, Python 3.10, SQLAlchemy 2.0, Pydantic 2.0
- **Database**: MariaDB 10.8 with connection pooling
- **Security**: JWT, CSRF protection, CORS, rate limiting
- **DevOps**: Docker, Docker Compose, GitHub Actions
- **Testing**: Pytest, Jest, React Testing Library

## 📦 Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- Node.js 18+ (for local development)
- Python 3.10+ (for local development)

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/MilesAheadToo/The_Beer_Game.git
   cd The_Beer_Game
   ```

2. Copy the example environment files:
   ```bash
   make init-env
   cp frontend/.env.example frontend/.env
   ```

3. Start the application using Docker Compose:
   ```bash
   docker-compose up -d --build
   ```

4. Initialize the database (first time only):
   ```bash
   docker-compose exec backend python -m app.db.init_db
   ```

5. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Admin Dashboard: http://localhost:3000/admin
   - API Documentation: http://localhost:8000/docs
   - Database Admin (phpMyAdmin): http://localhost:8080
     - Username: root
     - Password: 19890617

## 🏗 Project Structure


```
beer-game/
├── backend/               # Python FastAPI backend
│   ├── app/
│   │   ├── api/          # API endpoints
│   │   ├── core/         # Core application logic
│   │   ├── db/           # Database models and migrations
│   │   ├── models/       # Pydantic models
│   │   └── services/     # Business logic and services
│   ├── requirements.txt  # Python dependencies
│   └── main.py          # FastAPI application entry point
├── frontend/            # React frontend
│   ├── public/
│   └── src/
│       ├── components/   # React components
│       ├── pages/        # Page components
│       ├── services/     # API services
│       └── App.js        # Main React component
└── docker-compose.yml   # Docker configuration
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Node.js (for frontend development)
- Python 3.8+

### Installation

1. Clone the repository
2. Set up environment variables (`make init-env` copies `.env.example` to `.env` or, if present, copies `.env.<hostname>` for machine-specific settings)
3. Run `docker-compose up --build`

### Development

- Backend: `cd backend && uvicorn main:app --reload`
- Frontend: `cd frontend && npm start`

## License

MIT
