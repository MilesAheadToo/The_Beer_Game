#!/bin/bash

# Set database environment variables
export MYSQL_USER=beer_user
export MYSQL_PASSWORD=Daybreak@2025
export MYSQL_HOST=db
export MYSQL_DB=beer_game

# Activate virtual environment
source venv/bin/activate

# Run the data generation script
python -m app.data.generate_training_data

echo "Data generation complete!"
