#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <topics> <text>"
    exit 1
fi

# Run the Python script
python3 huggingface.py "$1" "$2"
