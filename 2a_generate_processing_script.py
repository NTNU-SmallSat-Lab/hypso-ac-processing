#!/usr/bin/env python3

import os

# Path to the base directory
base_dir = "/home/camerop/HYPSO_DATA_SNR"
# Output file for commands
output_file = "commands.sh"

script_dir = os.path.dirname(os.path.realpath(__file__))
script = os.path.join(script_dir, "2b_process_capture.py")

print(f"Script directory: {script_dir}")

# Open the output file for writing
with open(output_file, 'w') as f:
    # Write Bash header and setup logging environment
    f.write("#!/bin/bash\n\n")
    f.write("# Enable Python fault handler to catch segfault lines\n")
    f.write("export PYTHONFAULTHANDLER=1\n\n")
    f.write("# Create a log directory\n")
    f.write("LOG_DIR=\"/home/camerop/AC/logs\"\n")
    f.write("mkdir -p \"$LOG_DIR\"\n\n")

    # Iterate over all entries in the base directory
    for entry in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, entry)

        # Check if the entry is a directory
        if os.path.isdir(full_path):
            f.write(f"echo \"=========================================\"\n")
            f.write(f"echo \"Processing: {entry}\"\n")
            f.write(f"echo \"=========================================\"\n")
            
            # Construct the execution command with output logging
            log_file = f"\"$LOG_DIR/{entry}.log\""
            command_str = f"python {script} {full_path} > {log_file} 2>&1"
            f.write(command_str + '\n')
            
            # Append Bash logic to handle a crash and keep moving
            f.write("if [ $? -ne 0 ]; then\n")
            f.write(f"    echo \"❌ FAILED: {entry} crashed (Check log for details)\"\n")
            f.write(f"    echo \"{entry}\" >> \"$LOG_DIR/failed_captures.txt\"\n")
            f.write("else\n")
            f.write(f"    echo \"✅ SUCCESS: {entry} completed.\"\n")
            f.write("fi\n\n")
            
            print(f"Writing command block for: {entry}")

print(f"\nRobust commands file has been written to {output_file}")
