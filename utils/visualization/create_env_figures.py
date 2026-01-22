"""
Create publication-quality figures for RedBlueButton and Overcooked environments.

This script uses:
- Existing RedBlueButton visualization code for the RedBlueButton environment
- Overcooked's built-in StateVisualizer for the Overcooked environment
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Add overcooked_ai src to path
overcooked_src = project_root / "environments" / "overcooked_ai" / "src"
sys.path.insert(0, str(overcooked_src))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Overcooked imports are done lazily in create_overcooked_figure() to avoid import errors


def create_redbluebutton_figure(output_path=None, dpi=300):
    """
    Create a MiniGrid-style visualization of the RedBlueButton environment.
    Matches MiniGrid's dark grid aesthetic with light gray lines and simple colored shapes.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    # Grid dimensions
    width = 3
    height = 3
    
    # MiniGrid-style colors: dark cells with light grid lines
    bg_color = '#808080'  # Medium gray background (like MiniGrid border)
    cell_color = '#000000'  # Black cells (MiniGrid floor)
    grid_line_color = '#CCCCCC'  # Light gray grid lines
    
    # Set up grid
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    # Draw uniform grid cells (MiniGrid style: black cells with light gray borders)
    for x in range(width):
        for y in range(height):
            rect = mpatches.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                     facecolor=cell_color, 
                                     edgecolor=grid_line_color, 
                                     linewidth=1.0,
                                     zorder=0)
            ax.add_patch(rect)
    
    # Example configuration
    agent_pos = (0, 0)  # Bottom-left
    red_button_pos = (1, 1)  # Center
    blue_button_pos = (2, 2)  # Top-right
    
    # Draw agent as robot (rectangular body with square head)
    # Robot body (main rectangle)
    robot_body = mpatches.Rectangle((agent_pos[0] - 0.25, agent_pos[1] - 0.2), 
                                    0.5, 0.4,
                                    facecolor='#FFD700',  # Gold/yellow
                                    edgecolor='#CCAA00', 
                                    linewidth=0.5,
                                    zorder=3)
    ax.add_patch(robot_body)
    
    # Robot head (smaller square on top)
    robot_head = mpatches.Rectangle((agent_pos[0] - 0.15, agent_pos[1] - 0.35), 
                                    0.3, 0.15,
                                    facecolor='#FFD700',  # Gold/yellow
                                    edgecolor='#CCAA00', 
                                    linewidth=0.5,
                                    zorder=3)
    ax.add_patch(robot_head)
    
    # Robot eyes (two small circles)
    eye1 = mpatches.Circle((agent_pos[0] - 0.08, agent_pos[1] - 0.28), 
                          0.03,
                          facecolor='#000000',  # Black eyes
                          edgecolor='#000000', 
                          linewidth=0.3,
                          zorder=4)
    ax.add_patch(eye1)
    
    eye2 = mpatches.Circle((agent_pos[0] + 0.08, agent_pos[1] - 0.28), 
                          0.03,
                          facecolor='#000000',  # Black eyes
                          edgecolor='#000000', 
                          linewidth=0.3,
                          zorder=4)
    ax.add_patch(eye2)
    
    # Robot hands (small rectangles on sides)
    hand_left = mpatches.Rectangle((agent_pos[0] - 0.35, agent_pos[1] - 0.1), 
                                   0.1, 0.2,
                                   facecolor='#FFD700',  # Gold/yellow
                                   edgecolor='#CCAA00', 
                                   linewidth=0.5,
                                   zorder=3)
    ax.add_patch(hand_left)
    
    hand_right = mpatches.Rectangle((agent_pos[0] + 0.25, agent_pos[1] - 0.1), 
                                    0.1, 0.2,
                                    facecolor='#FFD700',  # Gold/yellow
                                    edgecolor='#CCAA00', 
                                    linewidth=0.5,
                                    zorder=3)
    ax.add_patch(hand_right)
    
    # Robot feet (small rectangles at bottom)
    foot_left = mpatches.Rectangle((agent_pos[0] - 0.2, agent_pos[1] + 0.2), 
                                   0.15, 0.1,
                                   facecolor='#FFD700',  # Gold/yellow
                                   edgecolor='#CCAA00', 
                                   linewidth=0.5,
                                   zorder=3)
    ax.add_patch(foot_left)
    
    foot_right = mpatches.Rectangle((agent_pos[0] + 0.05, agent_pos[1] + 0.2), 
                                    0.15, 0.1,
                                    facecolor='#FFD700',  # Gold/yellow
                                    edgecolor='#CCAA00', 
                                    linewidth=0.5,
                                    zorder=3)
    ax.add_patch(foot_right)
    
    # Draw red button as circle (MiniGrid style: flat bright red)
    red_circle = mpatches.Circle((red_button_pos[0], red_button_pos[1]), 
                                 0.35,
                                 facecolor='#FF0000',  # Bright red
                                 edgecolor='#FF0000', 
                                 linewidth=0.5,
                                 zorder=3)
    ax.add_patch(red_circle)
    
    # Draw blue button as circle (MiniGrid style: flat bright blue)
    blue_circle = mpatches.Circle((blue_button_pos[0], blue_button_pos[1]), 
                                  0.35,
                                  facecolor='#0000FF',  # Bright blue
                                  edgecolor='#0000FF', 
                                  linewidth=0.5,
                                  zorder=3)
    ax.add_patch(blue_circle)
    
    # Set background color (gray border like MiniGrid)
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)
    
    # Set axis properties - clean, minimal
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor=bg_color, edgecolor='none')
        print(f"✓ Saved RedBlueButton figure to: {output_path}")
    else:
        plt.savefig(project_root / 'results' / 'redbluebutton_env.png', 
                   dpi=dpi, bbox_inches='tight', facecolor=bg_color, edgecolor='none')
        print(f"✓ Saved RedBlueButton figure to: results/redbluebutton_env.png")
    
    return fig


def create_overcooked_figure(output_path=None, dpi=300):
    """
    Create a clean visualization of the Overcooked cramped_room layout.
    Uses Overcooked's built-in StateVisualizer.
    """
    # Lazy imports to avoid errors when only generating RedBlueButton
    try:
        # Initialize pygame before importing StateVisualizer (it loads images at class definition time)
        import pygame
        import os
        # Set SDL environment variables for headless operation
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        # Create a dummy display surface (needed for some pygame operations)
        pygame.display.set_mode((1, 1))
    except Exception as e:
        print(f"Warning: Could not initialize pygame: {e}")
        raise
    
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
    
    # Load the cramped_room layout
    mdp = OvercookedGridworld.from_layout_name(layout_name="cramped_room")
    
    # Get initial state
    initial_state = mdp.get_standard_start_state()
    
    # Create visualizer
    visualizer = StateVisualizer()
    
    # Render the state to an image
    # The render_state method returns a pygame surface which we need to convert
    try:
        import pygame
        
        # Render state
        surface = visualizer.render_state(
            initial_state,
            mdp.terrain_mtx,
            hud_data=None
        )
        
        # Convert pygame surface to numpy array
        pygame_image = pygame.surfarray.array3d(surface)
        pygame_image = np.transpose(pygame_image, (1, 0, 2))
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(pygame_image)
        ax.axis('off')
        
        # Add title
        ax.set_title('Overcooked Environment (cramped_room layout)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add task description
        ax.text(0.5, 0.02, 
               'Goal: Gather ingredients → Cook soup → Serve', 
               ha='center', va='bottom', 
               transform=fig.transFigure,
               fontsize=11, style='italic', color='#555555')
        
        plt.tight_layout()
        
        # Save final figure
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved Overcooked figure to: {output_path}")
        else:
            plt.savefig(project_root / 'results' / 'overcooked_env.png', 
                       dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved Overcooked figure to: results/overcooked_env.png")
        
        return fig
        
    except Exception as e:
        print(f"Error creating Overcooked visualization: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Create both environment figures."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create environment visualization figures')
    parser.add_argument('--redbluebutton', action='store_true', 
                       help='Create RedBlueButton figure')
    parser.add_argument('--overcooked', action='store_true', 
                       help='Create Overcooked figure')
    parser.add_argument('--both', action='store_true', default=True,
                       help='Create both figures (default)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: results/)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Resolution in DPI (default: 300)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else project_root / 'results'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.both or args.redbluebutton:
        rbb_path = output_dir / 'redbluebutton_env.png'
        create_redbluebutton_figure(output_path=rbb_path, dpi=args.dpi)
        plt.close('all')
    
    if args.both or args.overcooked:
        oc_path = output_dir / 'overcooked_env.png'
        create_overcooked_figure(output_path=oc_path, dpi=args.dpi)
        plt.close('all')
    
    print(f"\n{'='*60}")
    print("Environment figures created successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
