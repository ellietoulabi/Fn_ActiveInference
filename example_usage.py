from generative_models.SA_ActiveInference.RedBlueButton.B import B_fn
import numpy as np
import jax.numpy as jnp


# Grid params
width = 3
height = 3
S = width * height

# Action names for readable output
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "OPEN", 5: "NOOP"}

print("="*70)
print("Testing B with position-dependent button pressing")
print("="*70)

# Test 1: Agent NOT at button, tries to press
print("\n" + "="*70)
print("Test 1: Agent at pos 4, red button at pos 2 (NOT collocated)")
print("="*70)
qs_not_at_button = {
    "agent_pos": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),  # agent at 4
    "red_button_pos": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # button at 2
    "blue_button_pos": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  # button at 6
    "red_button_state": np.array([1.0, 0.0]),  # not pressed
    "blue_button_state": np.array([1.0, 0.0]),  # not pressed
}

qs_next = B_fn(qs_not_at_button, action=4, width=width, height=height, B_NOISE_LEVEL=0.05)
print(f"\nAfter OPEN action:")
print(f"  Red button: not_pressed={qs_next['red_button_state'][0]:.3f}, pressed={qs_next['red_button_state'][1]:.3f}")
print(f"  Blue button: not_pressed={qs_next['blue_button_state'][0]:.3f}, pressed={qs_next['blue_button_state'][1]:.3f}")
print(f"  Expected: Both should stay NOT pressed (only noise flipping)")

# Test 2: Agent AT red button, tries to press
print("\n" + "="*70)
print("Test 2: Agent at pos 2, red button at pos 2 (COLLOCATED)")
print("="*70)
qs_at_red = {
    "agent_pos": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # agent at 2
    "red_button_pos": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # button at 2
    "blue_button_pos": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  # button at 6
    "red_button_state": np.array([1.0, 0.0]),  # not pressed
    "blue_button_state": np.array([1.0, 0.0]),  # not pressed
}

qs_next = B_fn(qs_at_red, action=4, width=width, height=height, B_NOISE_LEVEL=0.05)
print(f"\nAfter OPEN action:")
print(f"  Red button: not_pressed={qs_next['red_button_state'][0]:.3f}, pressed={qs_next['red_button_state'][1]:.3f}")
print(f"  Blue button: not_pressed={qs_next['blue_button_state'][0]:.3f}, pressed={qs_next['blue_button_state'][1]:.3f}")
print(f"  Expected: Red should be PRESSED (80%), Blue stays not pressed")

# Test 3: Agent AT blue button, tries to press
print("\n" + "="*70)
print("Test 3: Agent at pos 6, blue button at pos 6 (COLLOCATED)")
print("="*70)
qs_at_blue = {
    "agent_pos": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  # agent at 6
    "red_button_pos": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # button at 2
    "blue_button_pos": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  # button at 6
    "red_button_state": np.array([1.0, 0.0]),  # not pressed
    "blue_button_state": np.array([1.0, 0.0]),  # not pressed
}

qs_next = B_fn(qs_at_blue, action=4, width=width, height=height, B_NOISE_LEVEL=0.05)
print(f"\nAfter OPEN action:")
print(f"  Red button: not_pressed={qs_next['red_button_state'][0]:.3f}, pressed={qs_next['red_button_state'][1]:.3f}")
print(f"  Blue button: not_pressed={qs_next['blue_button_state'][0]:.3f}, pressed={qs_next['blue_button_state'][1]:.3f}")
print(f"  Expected: Blue should be PRESSED (80%), Red stays not pressed")

# Test 4: Uncertain position (50-50 at button or not)
print("\n" + "="*70)
print("Test 4: Agent uncertain: 50% at pos 2, 50% at pos 4")
print("="*70)
qs_uncertain = {
    "agent_pos": np.array([0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),  # 50% at 2, 50% at 4
    "red_button_pos": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # button at 2
    "blue_button_pos": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  # button at 6
    "red_button_state": np.array([1.0, 0.0]),  # not pressed
    "blue_button_state": np.array([1.0, 0.0]),  # not pressed
}

qs_next = B_fn(qs_uncertain, action=4, width=width, height=height, B_NOISE_LEVEL=0.05)
print(f"\nAfter OPEN action:")
print(f"  Red button: not_pressed={qs_next['red_button_state'][0]:.3f}, pressed={qs_next['red_button_state'][1]:.3f}")
print(f"  Expected: Red ~50% chance to press → ~40% pressed (0.5 * 0.8)")

# Test 5: Already pressed button
print("\n" + "="*70)
print("Test 5: Agent at pos 2, red button already pressed")
print("="*70)
qs_already_pressed = {
    "agent_pos": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # agent at 2
    "red_button_pos": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # button at 2
    "blue_button_pos": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  # button at 6
    "red_button_state": np.array([0.0, 1.0]),  # already pressed
    "blue_button_state": np.array([1.0, 0.0]),  # not pressed
}

qs_next = B_fn(qs_already_pressed, action=4, width=width, height=height, B_NOISE_LEVEL=0.05)
print(f"\nAfter OPEN action:")
print(f"  Red button: not_pressed={qs_next['red_button_state'][0]:.3f}, pressed={qs_next['red_button_state'][1]:.3f}")
print(f"  Expected: Red stays PRESSED (100%)")

print("\n" + "="*70)
print("Test complete!")
print("="*70)
