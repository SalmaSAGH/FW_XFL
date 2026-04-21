#!/usr/bin/env python3
"""Fix data-dir path in run_client_raspberry_pi.py"""

import re
import sys

file_path = sys.argv[1] if len(sys.argv) > 1 else '/home/pi1/XFL-RPiLab/client/run_client_raspberry_pi.py'

with open(file_path, 'r') as f:
    content = f.read()

# Replace the default data-dir
content = re.sub(r"default='/app/data'", "default='/home/pi1/XFL-RPiLab/data'", content)

with open(file_path, 'w') as f:
    f.write(content)

print("Updated data-dir default")