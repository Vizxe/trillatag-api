#!/bin/bash

# Function to handle keyboard interrupt
function ctrl_c {
    echo -e "\nKilling container!"
    # Add your cleanup actions here
    exit 0
}
# Register the keyboard interrupt handler
trap ctrl_c SIGTERM SIGINT SIGQUIT SIGHUP

# Assemble CMD and extra launch args
eval "extra_launch_args=($EXTRA_LAUNCH_ARGS)"
LAUNCHER=($@ ${extra_launch_args[@]})

# Launch the server with ${CMD[@]} + ${EXTRA_LAUNCH_ARGS[@]}
"${LAUNCHER[@]}" &

python app/api_app.py
