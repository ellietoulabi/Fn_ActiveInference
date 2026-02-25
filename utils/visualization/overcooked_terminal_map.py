"""
Terminal renderer for Overcooked cramped_room using a small, aligned Unicode grid.

Public API:
    render_overcooked_grid(state, model_init) -> list[str]
        Returns a list of text lines representing the current map.

    orientation_str(player) -> tuple[str, str]
        Returns (\"NORTH\"/..., arrow_glyph) for a PlayerState.
"""

from __future__ import annotations

from typing import List, Tuple

try:
    # For correct visual width of Unicode symbols in terminal
    from wcwidth import wcswidth as _wcswidth
except ImportError:  # graceful fallback if wcwidth not installed
    def _wcswidth(s: str) -> int:  # type: ignore[override]
        return len(s)


_ORIENTATION_ARROWS = {
    (0, -1): "\u25B2",  # ▲ NORTH
    (0, 1): "\u25BC",   # ▼ SOUTH
    (1, 0): "\u25B6",   # ▶ EAST
    (-1, 0): "\u25C0",  # ◀ WEST
}


def orientation_str(player) -> Tuple[str, str]:
    """
    Return (name, arrow) from player.orientation (dx, dy).

    Name is one of {NORTH,SOUTH,EAST,WEST}, arrow is one of {▲,▼,▶,◀}.
    """
    orient = getattr(player, "orientation", (0, -1))
    arrow = _ORIENTATION_ARROWS.get(orient, "?")
    names = {(0, -1): "NORTH", (0, 1): "SOUTH", (1, 0): "EAST", (-1, 0): "WEST"}
    return names.get(orient, "?"), arrow


def render_overcooked_grid(state, model_init, cell_width: int = 2) -> List[str]:
    """
    Render a compact, single-line-per-row grid using Unicode symbols.

    Legend:
      Pot states: ⓪①②③❸
      Counters:  🅲/🅾/🅿/🆂  (empty/onion/plate/soup)
      Dispensers: 🄾 onion, 🄳 plate
      Serving:   🅂
      Agents:    ▲▼◀▶ (orientation), color-coded (A0 yellow, A1 orange)
    """
    CELL_WIDTH = cell_width

    w, h = model_init.GRID_WIDTH, model_init.GRID_HEIGHT

    # Objects and agents by position
    obj_at = {pos: obj for (pos, obj) in state.objects.items() if obj}
    agents = {p.position: (i, getattr(p, "orientation", (0, -1)))
              for i, p in enumerate(state.players)}

    def pot_glyph(x: int, y: int) -> str:
        glyph_empty = "⓪"
        obj = obj_at.get((x, y))
        if not obj or getattr(obj, "name", None) != "soup":
            return glyph_empty
        ingredients = getattr(obj, "_ingredients", None) or getattr(obj, "ingredients", [])
        onion_count = 0
        for ing in ingredients:
            ing_name = ing if isinstance(ing, str) else getattr(ing, "name", None)
            if ing_name == "onion":
                onion_count += 1
        is_ready = bool(getattr(obj, "is_ready", False))
        if is_ready:
            return "❸"
        if onion_count == 0:
            return "⓪"
        if onion_count == 1:
            return "①"
        if onion_count == 2:
            return "②"
        if onion_count >= 3:
            return "③"
        return glyph_empty

    def base_symbol(x: int, y: int) -> str:
        """Symbol for terrain/objects, without agents."""
        grid_idx = model_init.xy_to_index(x, y)
        if grid_idx in model_init.POT_INDICES:
            return pot_glyph(x, y)
        if grid_idx in model_init.COUNTER_INDICES:
            obj = obj_at.get((x, y))
            name = getattr(obj, "name", None) if obj else None
            if name == "onion":
                return "🅾"
            if name in ("dish", "plate"):
                return "🅿"
            if name == "soup":
                return "🆂"
            return "🅲"
        if grid_idx in model_init.SERVING_INDICES:
            return "🅂"
        if grid_idx in model_init.ONION_DISPENSER_INDICES:
            return "🄾"
        if grid_idx in model_init.DISH_DISPENSER_INDICES:
            return "🄳"
        return " "

    def pad_cell(sym: str) -> str:
        """Pad symbol to CELL_WIDTH based on visual width."""
        w_vis = _wcswidth(sym)
        if w_vis < 0:
            w_vis = len(sym)
        if w_vis >= CELL_WIDTH:
            return sym
        return sym + " " * (CELL_WIDTH - w_vis)

    arrow_for_dir = {
        (0, -1): "▲",
        (0, 1): "▼",
        (1, 0): "▶",
        (-1, 0): "◀",
    }

    def color_arrow(agent_idx: int, plain: str) -> str:
        padded = pad_cell(plain)
        if agent_idx == 0:
            return f"\033[93m{padded}\033[0m"
        return f"\033[38;5;208m{padded}\033[0m"

    lines: List[str] = []
    for y in range(h):
        row = ""
        for x in range(w):
            pos = (x, y)
            if pos in agents:
                agent_idx, orient = agents[pos]
                arrow = arrow_for_dir.get(orient, "▲")
                cell = color_arrow(agent_idx, arrow)
            else:
                cell = pad_cell(base_symbol(x, y))
            row += cell
        lines.append(row)

    return lines

