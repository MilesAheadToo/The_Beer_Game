# The Beer Game with Amazon Supply Chain Model

A simulation of the Beer Distribution Game using Amazon's Supply Chain Data Model, featuring:
- MariaDB database integration
- Graph Neural Network (GNN) for supply chain analysis
- Smart agents at each supply chain node
- Configurable distributions for lead times, capacities, and throughput
- React-based web interface with Material-UI
- Containerized with Docker for easy deployment

## 🚀 Features

- **Interactive Supply Chain Visualization**: Drag-and-drop interface for building supply chain networks
- **Advanced Simulation Engine**: Configurable parameters for demand, lead times, and inventory policies
- **Real-time Analytics**: Monitor key performance indicators and metrics
- **Bullwhip Effect Analysis**: Visualize and analyze demand amplification in the supply chain
- **Role-based Access Control**: Secure authentication and authorization
- **API-First Design**: RESTful API for integration with other systems

## 🛠 Tech Stack

- **Frontend**: React, Material-UI, Redux, React Flow, Recharts
- **Backend**: FastAPI, Python 3.9
- **Database**: MariaDB
- **AI/ML**: PyTorch, PyTorch Geometric
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions (sample configuration included)

## 📦 Prerequisites

- Docker 20.10+ and Docker Compose 1.29+
- Node.js 16+ (for local development)
- Python 3.9+ (for local development)

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/beer-game.git
   cd beer-game
   ```

2. Start the application using Docker Compose:
   ```bash
   docker-compose up -d
   ```

3. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
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
2. Set up environment variables (copy `.env.example` to `.env` and configure)
3. Run `docker-compose up --build`

### Development

- Backend: `cd backend && uvicorn main:app --reload`
- Frontend: `cd frontend && npm start`

## License

MIT
