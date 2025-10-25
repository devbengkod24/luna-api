#!/bin/bash
# Script to reset database (WARNING: This will delete all data!)

read -p "Are you sure you want to reset the database? This will delete all data! (yes/no): " confirm

if [ "$confirm" = "yes" ]; then
    echo "Downgrading database..."
    alembic downgrade base
    
    echo "Running migrations..."
    alembic upgrade head
    
    echo "Database reset complete!"
else
    echo "Reset cancelled."
fi

