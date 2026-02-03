"""
Test that cramped_room.layout's cook_time is applied when loading the layout.

Run from project root:
  python environments/test_cramped_room_cook_time.py

Or with pytest:
  pytest environments/test_cramped_room_cook_time.py -v
"""
import sys
from pathlib import Path

# Add overcooked_ai src so we can import overcooked_ai_py
project_root = Path(__file__).resolve().parent.parent
overcooked_src = project_root / "overcooked_ai" / "src"
if not overcooked_src.exists():
    overcooked_src = project_root / "environments" / "overcooked_ai" / "src"
if str(overcooked_src) not in sys.path:
    sys.path.insert(0, str(overcooked_src))

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Recipe, SoupState


def test_cramped_room_cook_time_from_layout():
    """Load cramped_room layout and assert cook_time from layout is 0."""
    print("\n--- test_cramped_room_cook_time_from_layout ---")
    print("Step 1: Loading layout 'cramped_room'...")
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    print("        MDP created.")

    # Recipe is configured from layout; cook_time should be in recipe_config
    print("Step 2: Check mdp.recipe_config has 'cook_time'...")
    assert "cook_time" in mdp.recipe_config, "layout should pass cook_time into recipe config"
    print(f"        recipe_config['cook_time'] = {mdp.recipe_config['cook_time']}")
    assert mdp.recipe_config["cook_time"] == 0, (
        f"expected cook_time=0 from cramped_room.layout, got {mdp.recipe_config['cook_time']}"
    )
    print("        OK: cook_time is 0.")

    # Recipe class should report the same
    print("Step 3: Check Recipe.configuration['cook_time']...")
    assert Recipe.configuration["cook_time"] == 0, (
        f"Recipe.configuration['cook_time'] should be 0, got {Recipe.configuration['cook_time']}"
    )
    print(f"        Recipe.configuration['cook_time'] = {Recipe.configuration['cook_time']}")
    print("        OK.")

    # Any recipe's .time should be 0 (used by soups)
    print("Step 4: Check every recipe's .time is 0...")
    for recipe in Recipe.ALL_RECIPES:
        assert recipe.time == 0, f"recipe {recipe} should have time=0, got {recipe.time}"
        print(f"        recipe {recipe.ingredients} -> time = {recipe.time}")
    print("        OK: all recipes have time=0.")


def test_soup_ready_immediately_with_cook_time_zero():
    """With cook_time=0, soup should be ready as soon as cooking begins."""
    print("\n--- test_soup_ready_immediately_with_cook_time_zero ---")
    print("Step 1: Load layout and ensure cook_time=0...")
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    assert Recipe.configuration["cook_time"] == 0
    print(f"        Recipe.configuration['cook_time'] = {Recipe.configuration['cook_time']}")

    # Get a pot position and create a soup with 3 onions (full), then begin cooking
    print("Step 2: Get standard start state and empty pot positions...")
    state = mdp.get_standard_start_state()
    pot_positions = mdp.get_pot_states(state)["empty"]
    assert pot_positions, "cramped_room should have at least one pot"
    pot_pos = pot_positions[0]
    print(f"        pot position = {pot_pos}")

    print("Step 3: Create soup (3 onions, cooking_tick=-1 = idle)...")
    soup = SoupState.get_soup(
        pot_pos,
        num_onions=3,
        num_tomatoes=0,
        cooking_tick=-1,
    )
    print(f"        soup.is_idle = {soup.is_idle}, soup.is_ready = {soup.is_ready}, soup.cook_time = {soup.cook_time}")
    assert soup.is_idle
    assert not soup.is_ready
    print("        OK: soup is idle and not ready.")

    print("Step 4: Call soup.begin_cooking()...")
    soup.begin_cooking()
    print(f"        soup.is_ready = {soup.is_ready}, soup.cook_time_remaining = {soup.cook_time_remaining}")
    # With cook_time=0: cooking_tick is 0, cook_time is 0, so 0 >= 0 -> ready
    assert soup.is_ready, "with cook_time=0, soup should be ready immediately after begin_cooking()"
    assert soup.cook_time_remaining == 0
    print("        OK: soup is ready immediately with no extra cook steps.")


if __name__ == "__main__":
    print("Running cook_time tests (step-by-step logs below).")
    test_cramped_room_cook_time_from_layout()
    print("\n  test_cramped_room_cook_time_from_layout passed.")

    test_soup_ready_immediately_with_cook_time_zero()
    print("\n  test_soup_ready_immediately_with_cook_time_zero passed.")

    print("\n========== All tests passed: cook_time=0 from cramped_room.layout is working. ==========")
