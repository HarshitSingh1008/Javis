import re

# Read the original file
with open('E:/Jarvis 2.0/venv/nexus_brain/orchestrator.py', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# 1. Add concurrency control fields to __init__
init_pattern = r'(        # Queue for UI streaming\s+self\._output_queue: Optional\[asyncio\.Queue\] = None\s+self\._progress_queue: Optional\[asyncio\.Queue\] = None)'
init_replacement = r'''\1
        
        # Concurrency control
        self._run_lock = asyncio.Lock()
        self._max_concurrent_runs = 3
        self._active_runs = 0'''

content = re.sub(init_pattern, init_replacement, content)

# 2. Modify the run method to add concurrency control
# Find the run method docstring and add note
run_docstring_pattern = r'(        """\s+Run a complete agent cycle from input to response\.\s+.*?\s+        Returns:\s+            Final NexusState with response, metrics, and execution record\.\s+        )"""'
run_docstring_replacement = r'''\1        
        Note:
            This method is thread-safe for concurrent calls up to max_concurrent_runs.
            Excess calls will wait for a slot to become available.
        """'''

content = re.sub(run_docstring_pattern, run_docstring_replacement, content, flags=re.DOTALL)

# 3. Add concurrency control at start of run method
run_start_pattern = r'(        start = time\.perf_counter\(\)\s*\n\s*# Create initial state)'
run_start_replacement = r'''        start = time.perf_counter()
        
        # Concurrency control - acquire slot
        async with self._run_lock:
            while self._active_runs >= self._max_concurrent_runs:
                await asyncio.sleep(0.1)
            self._active_runs += 1
        
        try:
            # Create initial state'''

content = re.sub(run_start_pattern, run_start_replacement, content)

# 4. Add finally block before return state at end of run method
run_end_pattern = r'(            return state\s*\n\s*)    # \xe2\x94\x80\xe2\x94\x80 Graph Node'
run_end_replacement = r'''            return state
        finally:
            # Release concurrency slot
            async with self._run_lock:
                self._active_runs = max(0, self._active_runs - 1)
    
    # \xe2\x94\x80\xe2\x94\x80 Graph Node'''

content = re.sub(run_end_pattern, run_end_replacement, content)

# Write the modified content
with open('E:/Jarvis 2.0/venv/nexus_brain/orchestrator.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Modifications applied successfully')