#!/bin/bash
# Script to create a new database migration

if [ -z "$1" ]; then
    echo "Usage: ./scripts/create_migration.sh \"migration description\""
    exit 1
fi

echo "Creating migration: $1"
alembic revision --autogenerate -m "$1"

