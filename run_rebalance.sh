#!/bin/bash

# Rebalance Bot Cron Wrapper Script
# This script ensures the correct Python environment is used when running from cron

# Set the project directory
PROJECT_DIR="/Users/mattnicolaysen/Documents/Projects/DeFi/Rebalance Bot - Coinbase"

# Set the Python executable (use your current Python installation)
PYTHON="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Run the rebalance bot
"$PYTHON" -m src.main --mode once
