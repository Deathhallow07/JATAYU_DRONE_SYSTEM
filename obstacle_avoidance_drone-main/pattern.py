#!/usr/bin/env python3
"""
The lawnmower survey geometry, shared by both stacks so they fly the SAME path.

    lawnmower.py     Nav2 stack   (NavigateToPose goals, map frame)
    oa_lawnmower.py  ArduPilot OA (GUIDED position targets, LOCAL_NED)

Coordinates are metres from the takeoff point: x = EAST, y = NORTH. That is
exactly Nav2's `map` frame here (tf_publisher.py roots it at the EKF origin),
and it is also MAV_FRAME_LOCAL_NED once you swap the axes -- north=y, east=x,
down=-alt. Same numbers reach both stacks with no datum conversion in between,
which is the whole point when comparing them.

Stdlib only, on purpose: oa_lawnmower.py must run without ROS.

KEEP IN STEP WITH worlds/lawnmower.sdf. Buildings are placed against these
exact numbers; changing one without the other makes the comparison meaningless.
"""

LANES = [-30.0, -15.0, 0.0, 15.0, 30.0]
Y0, Y1 = 5.0, 35.0

# Half-extent of the Nav2 global costmap (100 x 100 m, origin -50,-50).
# Lanes stop well inside it: a blocked waypoint near the edge lets the
# controller push the drone off the map, after which the planner cannot even
# resolve its own start pose and every later goal fails for the wrong reason.
MAP_EXTENT = 50.0

# Which corners are deliberately unreachable, and why. Used for reporting only.
BLOCKED = {
    (-15.0, 35.0): 'corner_nw sits on top of it (goal inside an obstacle)',
    (15.0, 5.0): 'courtyard_se walls it in (goal free, but no route)',
}


def build_pattern(lanes=None, y0=None, y1=None):
    """Corners of the boustrophedon, in flight order.

    Two per lane; every other lane is walked backwards so consecutive lanes
    join end-to-end instead of flying back to the start of each one.
    """
    lanes = LANES if lanes is None else lanes
    y0 = Y0 if y0 is None else y0
    y1 = Y1 if y1 is None else y1

    pts = []
    for i, x in enumerate(lanes):
        a, b = (y0, y1) if i % 2 == 0 else (y1, y0)
        pts.append((x, a))
        pts.append((x, b))
    return pts


def describe(pattern):
    """Human-readable listing, with the blocked corners called out."""
    out = []
    for i, (x, y) in enumerate(pattern, 1):
        note = BLOCKED.get((x, y), '')
        out.append(f'  {i:2d}  ({x:7.1f}, {y:7.1f})'
                   + (f'   BLOCKED: {note}' if note else ''))
    return '\n'.join(out)


if __name__ == '__main__':
    p = build_pattern()
    print(f'{len(p)} waypoints:')
    print(describe(p))
