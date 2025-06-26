#!/bin/bash

# Run the thrust addition test
echo "Running QForte Thrust Addition Test..."
python test_thrust_addition.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "Test completed successfully!"
else
    echo "Test failed with error code $?"
fi
