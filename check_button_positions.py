"""Check actual button positions in environment."""

from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv

env = SingleAgentRedBlueButtonEnv(width=3, height=3, max_steps=10)
env.reset()

print("Environment setup:")
print(f"  Red button position (x,y): {env.red_button}")
print(f"  Blue button position (x,y): {env.blue_button}")
print(f"  Agent start (x,y): {env.agent_position}")

print("\nGrid indexing (y * width + x):")
print("  Position 0 = (0,0) = top-left")
print("  Position 1 = (1,0)")
print("  Position 2 = (2,0) = top-right")
print("  Position 3 = (0,1)")
print("  Position 4 = (1,1)")
print("  Position 5 = (2,1)")
print("  Position 6 = (0,2) = bottom-left")
print("  Position 7 = (1,2)")
print("  Position 8 = (2,2) = bottom-right")

# Calculate indices
red_idx = env.red_button[1] * 3 + env.red_button[0]
blue_idx = env.blue_button[1] * 3 + env.blue_button[0]
agent_idx = env.agent_position[1] * 3 + env.agent_position[0]

print(f"\nConverted to indices:")
print(f"  Red button: index {red_idx}")
print(f"  Blue button: index {blue_idx}")
print(f"  Agent: index {agent_idx}")

print("\nGrid visualization:")
grid_vis = env.render(mode='silent')
for row in grid_vis:
    print(f"  {' '.join(row)}")
print("  (A=agent, r=red button, b=blue button)")

