"""Test if preferences are working."""

from generative_models.SA_ActiveInference.RedBlueButton import C, model_init

print("Testing C_fn (preferences)...")
print("="*70)

# Test different observation scenarios
test_cases = [
    {
        'name': 'Initial state (neutral)',
        'obs': {
            'agent_pos': 0,
            'on_red_button': 0,  # FALSE
            'on_blue_button': 0,  # FALSE
            'red_button_state': 0,  # not_pressed
            'blue_button_state': 0,  # not_pressed
            'game_result': 0,  # neutral
            'button_just_pressed': 0,  # FALSE
        }
    },
    {
        'name': 'On red button (not pressed)',
        'obs': {
            'agent_pos': 6,
            'on_red_button': 1,  # TRUE
            'on_blue_button': 0,  # FALSE
            'red_button_state': 0,  # not_pressed
            'blue_button_state': 0,  # not_pressed
            'game_result': 0,  # neutral
            'button_just_pressed': 0,  # FALSE
        }
    },
    {
        'name': 'Just pressed red button',
        'obs': {
            'agent_pos': 6,
            'on_red_button': 1,  # TRUE
            'on_blue_button': 0,  # FALSE
            'red_button_state': 1,  # pressed
            'blue_button_state': 0,  # not_pressed
            'game_result': 0,  # neutral
            'button_just_pressed': 1,  # TRUE
        }
    },
    {
        'name': 'Red pressed, now on blue',
        'obs': {
            'agent_pos': 2,
            'on_red_button': 0,  # FALSE
            'on_blue_button': 1,  # TRUE
            'red_button_state': 1,  # pressed
            'blue_button_state': 0,  # not_pressed
            'game_result': 0,  # neutral
            'button_just_pressed': 0,  # FALSE
        }
    },
    {
        'name': 'WIN! (red then blue)',
        'obs': {
            'agent_pos': 2,
            'on_red_button': 0,  # FALSE
            'on_blue_button': 1,  # TRUE
            'red_button_state': 1,  # pressed
            'blue_button_state': 1,  # pressed
            'game_result': 1,  # win
            'button_just_pressed': 1,  # TRUE
        }
    },
]

for test in test_cases:
    print(f"\n{test['name']}:")
    prefs = C.C_fn(test['obs'])
    
    total_pref = 0.0
    for mod, pref_val in prefs.items():
        print(f"  {mod:25s}: {pref_val:+.3f}")
        total_pref += pref_val
    
    print(f"  {'TOTAL':25s}: {total_pref:+.3f}")

print("\n" + "="*70)
print("Key observations:")
print("  - Total preference should increase as we get closer to goal")
print("  - WIN state should have highest total preference")
print("  - Button presses should give positive reward")


