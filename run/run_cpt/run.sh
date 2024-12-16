#!/bin/bash

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Kill Python processes in Windows
echo "Terminating Windows Python processes..."
tasklist | grep -i python | awk '{print $2}' | xargs -r -I {} taskkill //F //PID {}

# Kill Python processes in UCRT64/MSYS2
echo "Terminating UCRT64/MSYS2 Python processes..."
ps aux | grep python | grep -v grep | awk '{print $2}' | xargs -r kill -9

# Remove Old Files
echo "Removing old results..."
rm -rf ./*.xlsx
rm -rf ./*.html
rm -rf ./*.prof

# Start main.py
echo "Starting main.py..."
python main.py

# Check if main.py started successfully
if [ $? -eq 0 ]; then
    echo "main.py started successfully"
else
    echo "Error: Failed to start main.py"
    exit 1
fi