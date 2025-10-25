#!/bin/bash
# Script to run database migrations

echo "Running database migrations..."
alembic upgrade head
echo "Migrations complete!"

