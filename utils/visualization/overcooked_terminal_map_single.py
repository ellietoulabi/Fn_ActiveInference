"""
Single-agent terminal renderer for Overcooked cramped_room.

Same grid as overcooked_terminal_map but only draws the first player (agent 0).
Use this when running with a single controlled agent (e.g. IndividuallyCollective with dummy or cramped_room_single).

Public API:
    render_overcooked_grid_single(state, model_init, cell_width=2) -> list[str]
"""

from __future__ import annotations

from typing import List

try:
    from wcwidth import wcswidth as _wcswidth
except ImportError:
    def _wcswidth(s: str) -> int:
        return len(s)

from .overcooked_terminal_map import orientation_str  # noqa: F401


def render_overcooked_grid_single(state, model_init, cell_width: int = 2) -> List[str]:
    """
    Render the Overcooked grid showing only the first agent (agent 0).
    Other agents (e.g. a dummy) are not drawn so the map shows a single-agent view.
    """
    CELL_WIDTH = cell_width
    w, h = model_init.GRID_WIDTH, model_init.GRID_HEIGHT

    obj_at = {pos: obj for (pos, obj) in state.objects.items() if obj}
    # Only the first player
    agents = {}
    if state.players:
        p = state.players[0]
        agents[p.position] = (0, getattr(p, "orientation", (0, -1)))

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

    # Agent 0 only -> yellow
    def color_arrow(plain: str) -> str:
        padded = pad_cell(plain)
        return f"\033[93m{padded}\033[0m"

    lines: List[str] = []
    for y in range(h):
        row = ""
        for x in range(w):
            pos = (x, y)
            if pos in agents:
                _, orient = agents[pos]
                arrow = arrow_for_dir.get(orient, "▲")
                cell = color_arrow(arrow)
            else:
                cell = pad_cell(base_symbol(x, y))
            row += cell
        lines.append(row)

    return lines
