"""
Visualize the cramped_room layout for Overcooked using StateVisualizer.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Add overcooked_ai src to path
overcooked_src = project_root / "environments" / "overcooked_ai" / "src"
sys.path.insert(0, str(overcooked_src))

# Set SDL environment variables BEFORE any pygame imports
os.environ['SDL_VIDEODRIVER'] = 'dummy'

# Initialize pygame BEFORE importing StateVisualizer (images load at class definition time)
import pygame
pygame.init()
pygame.display.set_mode((1, 1))

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

def visualize_cramped_room(output_path=None, dpi=300):
    """Visualize the cramped_room layout using Overcooked's StateVisualizer."""
    # Load the cramped_room layout
    mdp = OvercookedGridworld.from_layout_name(layout_name="cramped_room")
    
    # Get initial state
    initial_state = mdp.get_standard_start_state()
    
    # Create visualizer
    visualizer = StateVisualizer()
    
    # Use display_rendered_state to save directly to file
    if output_path is None:
        output_path = str(project_root / 'results' / 'overcooked_cramped_room.png')
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    # Render and save
    visualizer.display_rendered_state(
        initial_state,
        mdp.terrain_mtx,
        hud_data=None,
        img_path=output_path,
        window_display=False,
        ipython_display=False
    )
    
    print(f"✓ Saved cramped_room visualization to: {output_path}")
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize cramped_room layout using StateVisualizer')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: results/overcooked_cramped_room.png)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Resolution in DPI (default: 300, note: pygame saves at native resolution)')
    
    args = parser.parse_args()
    
    visualize_cramped_room(output_path=args.output, dpi=args.dpi)
