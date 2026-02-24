#!/usr/bin/env python3

import os

# Path to the base directory
base_dir = "/home/camerop/HYPSO_DATA"
# Output file for commands
output_file = "commands.sh"

script_dir = os.path.dirname(os.path.realpath(__file__))
#script = os.path.join(script_dir, "2c_process_capture_6s.py")
script = os.path.join(script_dir, "2b_process_capture.py")

print(script_dir)

# Open the output file for writing
with open(output_file, 'w') as f:
    
    f.write("#!/bin/bash" + "\n")

    # Iterate over all entries in the base directory
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)

	# Check if the entry is a directory
        if os.path.isdir(full_path):
            # Construct the command
            command = ["python", str(script), full_path]
            command_str = ' '.join(command)
            
            # Write the command to the file
            f.write(command_str + '\n')
            
            print(f"Writing command: {command_str}")

print(f"\nCommands have been written to {output_file}")

