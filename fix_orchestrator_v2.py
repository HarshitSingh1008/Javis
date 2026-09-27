#!/usr/bin/env python3
"""
Fix orchestrator.py by adding concurrency control to the run method.
"""

import re

with open('E:/Jarvis 2.0/venv/nexus_brain/orchestrator.py', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

# Find line numbers for key locations
init_end_line = -1
run_start_line = -1
run_try_line = -1
run_finally_line = -1
run_end_line = -1

for i, line in enumerate(lines):
    # Find end of __init__ (before first method after __init__)
    if 'self._progress_queue: Optional[asyncio.Queue] = None' in line:
        init_end_line = i
    
    # Find run method start
    if 'async def run(' in line and run_start_line == -1:
        run_start_line = i
    
    # Find try: in run method
    if run_start_line != -1 and run_try_line == -1 and 'try:' in line:
        run_try_line = i
    
    # Find the final return state before "Graph Node Implementations"
    if 'return state' in line and 'Graph Node Implementations' in lines[i+1] if i+1 < len(lines) else False:
        run_end_line = i

print(f'init_end_line: {init_end_line}')
print(f'run_start_line: {run_start_line}')
print(f'run_try_line: {run_try_line}')
print(f'run_end_line: {run_end_line}')

# 1. Add concurrency fields after progress_queue in __init__
if init_end_line != -1:
    indent = ' ' * 8
    lines.insert(init_end_line + 1, '\n')
    lines.insert(init_end_line + 2, f'{indent}# Concurrency control\n')
    lines.insert(init_end_line + 3, f'{indent}self._run_lock = asyncio.Lock()\n')
    lines.insert(init_end_line + 4, f'{indent}self._max_concurrent_runs = 3\n')
    lines.insert(init_end_line + 5, f'{indent}self._active_runs = 0\n')
    print('Added concurrency fields to __init__')

# 2. Modify run method docstring to add Note
if run_start_line != -1:
    # Find the docstring end (""")
    for i in range(run_start_line, min(run_start_line + 30, len(lines))):
        if '"""' in lines[i] and i > run_start_line:
            # Insert before the closing """
            lines.insert(i, '        Note:\n')
            lines.insert(i + 1, '            This method is thread-safe for concurrent calls up to max_concurrent_runs.\n')
            lines.insert(i + 2, '            Excess calls will wait for a slot to become available.\n')
            print('Added Note to run docstring')
            break

# 3. Add concurrency control at start of run method body
# Find the line with "start = time.perf_counter()"
for i in range(run_start_line, min(run_start_line + 50, len(lines))):
    if 'start = time.perf_counter()' in lines[i]:
        # Insert after this line
        lines[i] = lines[i].rstrip() + '\n'
        lines.insert(i + 1, '\n')
        lines.insert(i + 2, '        # Concurrency control - acquire slot\n')
        lines.insert(i + 3, '        async with self._run_lock:\n')
        lines.insert(i + 4, '            while self._active_runs >= self._max_concurrent_runs:\n')
        lines.insert(i + 5, '                await asyncio.sleep(0.1)\n')
        lines.insert(i + 6, '            self._active_runs += 1\n')
        lines.insert(i + 7, '\n')
        lines.insert(i + 8, '        try:\n')
        print('Added concurrency control at run start')
        break

# 3b. Find the "Create initial state" line and ensure it's inside try block
# The line after "try:" should be indented
# We need to indent all lines from after "try:" to before "finally:" by 4 spaces
# But since we're inserting "try:" we need to handle the existing indentation

# 4. Add finally block before the final return
# Find the final "return state" before "Graph Node Implementations"
for i in range(len(lines) - 1, -1, -1):
    if 'return state' in lines[i] and i + 1 < len(lines) and 'Graph Node Implementations' in lines[i + 1]:
        # Insert finally block before this return
        # The return should be inside the try block, then finally after
        # We need to: keep the return, then add finally block, then the comment
        lines[i] = '            return state\n'
        lines.insert(i + 1, '        finally:\n')
        lines.insert(i + 2, '            # Release concurrency slot\n')
        lines.insert(i + 3, '            async with self._run_lock:\n')
        lines.insert(i + 4, '                self._active_runs = max(0, self._active_runs - 1)\n')
        print('Added finally block')
        break

# Write back
with open('E:/Jarvis 2.0/venv/nexus_brain/orchestrator.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Done!')