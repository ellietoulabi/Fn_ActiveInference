"""Profile A_fn performance."""

import time
import numpy as np
from generative_models.SA_ActiveInference.RedBlueButton import A, model_init

# Create a sample state
state_indices = {
    'agent_pos': 0,
    'red_button_pos': 2,
    'blue_button_pos': 6,
    'red_button_state': 0,
    'blue_button_state': 0,
}

# Warm up
for _ in range(10):
    A.A_fn(state_indices)

# Time it
num_calls = 1000
start = time.time()
for _ in range(num_calls):
    result = A.A_fn(state_indices)
elapsed = time.time() - start

print(f"A_fn performance:")
print(f"  {num_calls} calls in {elapsed*1000:.1f}ms")
print(f"  {elapsed/num_calls*1000:.3f}ms per call")
print(f"  {1.0/(elapsed/num_calls):.0f} calls/sec")


