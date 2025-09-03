#!/bin/bash
set -e

# Load environment variables
if [ -f .env.prod ]; then
    export $(grep -v '^#' .env.prod | xargs)
else
    echo "Error: .env.prod file not found!"
    exit 1
fi

# Create necessary directories
mkdir -p config/nginx/ssl

# Generate self-signed SSL certificate (replace with Let's Encrypt in production)
if [ ! -f config/nginx/ssl/cert.pem ] || [ ! -f config/nginx/ssl/key.pem ]; then
    echo "Generating self-signed SSL certificate..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout config/nginx/ssl/key.pem \
        -out config/nginx/ssl/cert.pem \
        -subj "/C=US/ST=State/L=City/O=BeerGame/CN=localhost"
    chmod 600 config/nginx/ssl/*.pem
fi

# Build frontend
echo "Building frontend..."
cd frontend
npm install
npm run build
cd ..

# Build and start services
echo "Starting production services..."
docker-compose -f docker-compose.prod.yml up -d --build

echo ""
echo "Production deployment complete!"
echo "Application should be available at https://localhost (accept the self-signed certificate)"
echo ""
echo "To view logs:"
echo "  docker-compose -f docker-compose.prod.yml logs -f"
echo ""
echo "To stop the application:"
echo "  docker-compose -f docker-compose.prod.yml down"
