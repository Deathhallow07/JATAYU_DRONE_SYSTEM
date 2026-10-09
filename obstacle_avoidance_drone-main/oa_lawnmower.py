#!/usr/bin/env python3
"""
Fly the lawnmower with ARDUPILOT's own obstacle avoidance (BendyRuler).

The counterpart to lawnmower.py. Same waypoints (pattern.py), same Gazebo
world, same drone -- but no ROS, no Nav2, no costmaps. ArduPilot plans around
obstacles itself using the proximity picture that mavlik_bridge.py feeds it as
MAVLink OBSTACLE_DISTANCE, with OA_TYPE=1 from oa.parm.

    python3 oa_lawnmower.py                    # fly it
    python3 oa_lawnmower.py --dry-run          # print waypoints, fly nothing

Assumes the drone is already airborne in GUIDED (start_sim_oa.sh does that with
auto_takeoff.py) and that mavlik_bridge.py is running.

ADVANCE RULE -- the goto_line.py algorithm
ArduPilot never reports "this destination is unreachable". BendyRuler simply
keeps steering around forever, so waiting for arrival would hang on a blocked
corner. goto_line.py's trick is used instead.

Note which line is watched: A is the waypoint currently being flown to, B is
the NEXT waypoint in the queue, and the offset measured is from the A->B line.
So while heading for A we are asking "am I already sitting on the run from A to
B?" -- and if avoidance has carried us onto it, there is nothing left to gain
by continuing to A, so we retarget B directly.

    offset > LINE_TOLERANCE   off the A->B line (the normal case on approach,
                              and what avoidance produces when it shoves us
                              sideways)
    ...then offset < TOL      now on the A->B line; skip A, go to B

goto_line.py requires the deviation FIRST for exactly this reason: a vehicle
that happened to start on the line would otherwise advance instantly. A leg
that ends with the vehicle actually arriving is caught by --arrive-radius, and
the final waypoint has no B, so it can only be reached or time out.

WHAT THIS DOES *NOT* DO, on purpose: it cannot tell "flew around it and
carried on" apart from "could never get there". Both simply advance. That is a
real difference from the Nav2 stack, where an unreachable goal produces an
explicit abort and shows up as SKIPPED. Compare the two summaries with that in
mind -- the Nav2 run classifies, this one only progresses.
"""

import argparse
import math
import sys
import time

from pymavlink import mavutil

from pattern import BLOCKED, build_pattern, describe

LINE_TOLERANCE = 2.0      # metres off the A-B line that counts as "deviated"
ARRIVE_RADIUS = 2.0       # metres from the target that counts as "reached"
LEG_TIMEOUT = 180.0       # backstop only; see --leg-timeout


def goto_ned(m, x_east, y_north, alt):
    """GUIDED position target in MAV_FRAME_LOCAL_NED.

    LOCAL_NED is rooted at the EKF origin, which is the takeoff point -- the
    same origin as Nav2's `map` frame. So the numbers in pattern.py go to both
    stacks untranslated. NED wants north/east/down, hence the axis swap and the
    negated altitude.
    """
    m.mav.set_position_target_local_ned_send(
        0, m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111111000,                    # position only
        y_north, x_east, -alt,
        0, 0, 0,
        0, 0, 0,
        0, 0)


def local_position(m, timeout=2.0):
    """Current (x_east, y_north, alt) from LOCAL_POSITION_NED, or None."""
    msg = m.recv_match(type='LOCAL_POSITION_NED', blocking=True,
                       timeout=timeout)
    if msg is None:
        return None
    return msg.y, msg.x, -msg.z


def cross_track(a, b, p):
    """Perpendicular distance from p to the infinite line a-b, in metres.

    Straight port of goto_line.py's version, minus the lat/lon projection --
    we are already in metres.
    """
    bx, by = b[0] - a[0], b[1] - a[1]
    px, py = p[0] - a[0], p[1] - a[1]
    span = math.hypot(bx, by)
    if span < 1e-6:
        return math.hypot(px, py)
    return abs(bx * py - by * px) / span


def fly_leg(m, target, nxt, alt, args):
    """Fly to `target`, watching the target->nxt line. goto_line.py's A and B.

    `nxt` is None for the final waypoint, which then has no line to watch and
    can only be REACHED or TIMEOUT.

    Returns (outcome, seconds, max_deviation).
    """
    goto_ned(m, target[0], target[1], alt)

    t0 = time.time()
    was_off_line = False
    max_dev = 0.0

    while time.time() - t0 < args.leg_timeout:
        pos = local_position(m)
        if pos is None:
            continue
        x, y, _ = pos

        if math.hypot(target[0] - x, target[1] - y) <= args.arrive_radius:
            return 'REACHED', time.time() - t0, max_dev

        if nxt is None:
            continue

        dev = cross_track(target, nxt, (x, y))
        max_dev = max(max_dev, dev)

        if dev > args.line_tolerance:
            was_off_line = True
        elif was_off_line:
            # On the A->B line having been off it. Nothing more to gain from
            # continuing to A, so retarget B. This is goto_line.py's condition.
            return 'ADVANCED (on A-B line)', time.time() - t0, max_dev

    return 'TIMEOUT', time.time() - t0, max_dev


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--connection', default='udpin:0.0.0.0:14557')
    ap.add_argument('--alt', type=float, default=10.0,
                    help='cruise altitude, m AGL (match the Nav2 run)')
    ap.add_argument('--line-tolerance', type=float, default=LINE_TOLERANCE)
    ap.add_argument('--arrive-radius', type=float, default=ARRIVE_RADIUS)
    ap.add_argument('--leg-timeout', type=float, default=LEG_TIMEOUT,
                    help='backstop so a wedged vehicle cannot hang the mission')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    pattern = build_pattern()
    if args.dry_run:
        print(f'{len(pattern)} waypoints:')
        print(describe(pattern))
        return 0

    print(f'connecting to {args.connection} ...')
    m = mavutil.mavlink_connection(args.connection)
    m.wait_heartbeat()
    print(f'heartbeat: system {m.target_system} component {m.target_component}')

    start = local_position(m, timeout=10.0)
    if start is None:
        print('no LOCAL_POSITION_NED -- is the drone armed and flying?',
              file=sys.stderr)
        return 1
    print(f'starting from ({start[0]:.1f}, {start[1]:.1f})')

    results = []
    try:
        for i, wp in enumerate(pattern, 1):
            # A = this waypoint, B = the next one in the queue.
            nxt = pattern[i] if i < len(pattern) else None
            tag = '  [blocked on purpose]' if tuple(wp) in BLOCKED else ''
            nxt_s = f'  -> next ({nxt[0]:.0f}, {nxt[1]:.0f})' if nxt else ''
            print(f'--- waypoint {i}/{len(pattern)}  '
                  f'({wp[0]:.1f}, {wp[1]:.1f}){tag}{nxt_s}')
            outcome, secs, dev = fly_leg(m, wp, nxt, args.alt, args)
            print(f'    {outcome} in {secs:.0f}s, max deviation {dev:.1f} m')
            results.append((i, wp[0], wp[1], outcome, secs, dev))
    except KeyboardInterrupt:
        print('\ninterrupted')
    finally:
        print('\n=== ardupilot OA lawnmower summary ===')
        for i, x, y, outcome, secs, dev in results:
            flag = ' *blocked' if (x, y) in BLOCKED else ''
            print(f'  {i:2d}  ({x:7.1f}, {y:7.1f})  {outcome:<20}'
                  f'{secs:5.0f}s  dev {dev:5.1f} m{flag}')
        reached = sum(1 for r in results if r[3] == 'REACHED')
        print(f'  {reached}/{len(results)} reached, '
              f'{len(results) - reached} skipped early via the A-B line rule')
    return 0


if __name__ == '__main__':
    sys.exit(main())
