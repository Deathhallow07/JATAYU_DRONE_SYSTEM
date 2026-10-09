#!/usr/bin/env python3
"""
Put SITL into GUIDED, arm, and climb to a target altitude -- no ROS, no mavros.

start_sim.sh runs this from pane 5 once mavros/Nav2 are up, so the drone is
already hovering by the time a Nav2 goal is sent. Run it by hand the same way:

    python3 auto_takeoff.py            # GUIDED, arm, takeoff to 10 m
    python3 auto_takeoff.py --alt 15

It listens on its own MAVLink endpoint (SITL's `--out=udp:127.0.0.1:14556`),
separate from mavros (14550) and the send_ned bridge (14555), so none of the
three steal each other's packets.
"""

import argparse
import sys
import time

from pymavlink import mavutil


def wait_ekf_ready(m, timeout):
    """Block until ArduPilot reports a usable position estimate.

    A cold SITL needs ~40-60 s for the EKF to settle and declare GPS lock;
    arming before that just trips a pre-arm check.
    """
    need = (mavutil.mavlink.EKF_ATTITUDE |
            mavutil.mavlink.EKF_VELOCITY_HORIZ |
            mavutil.mavlink.EKF_POS_HORIZ_REL |
            mavutil.mavlink.EKF_POS_HORIZ_ABS)
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = m.recv_match(type='EKF_STATUS_REPORT', blocking=True, timeout=5)
        if msg and (msg.flags & need) == need:
            return True
        print('  ... waiting for EKF (flags=%s)'
              % (hex(msg.flags) if msg else 'no report'))
    return False


def set_mode_confirmed(m, mode_name, timeout):
    """Request a mode and wait until HEARTBEAT actually reports it."""
    want = m.mode_mapping()[mode_name]
    deadline = time.time() + timeout
    while time.time() < deadline:
        m.set_mode(mode_name)
        end = time.time() + 2
        while time.time() < end:
            hb = m.recv_match(type='HEARTBEAT', blocking=True, timeout=2)
            if hb and hb.custom_mode == want:
                return True
        print(f'  ... {mode_name} not accepted yet, retrying')
    return False


def wait_armed(m, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        hb = m.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
        if hb and (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--connection', default='udpin:0.0.0.0:14556')
    ap.add_argument('--alt', type=float, default=10.0,
                    help='takeoff altitude in m AGL (default %(default)s)')
    ap.add_argument('--ekf-timeout', type=float, default=180.0)
    ap.add_argument('--mode-timeout', type=float, default=120.0,
                    help='seconds to keep retrying the GUIDED mode change')
    ap.add_argument('--climb-timeout', type=float, default=120.0)
    ap.add_argument('--stall-timeout', type=float, default=15.0,
                    help='seconds on the ground before re-sending takeoff')
    ap.add_argument('--takeoff-retries', type=int, default=3)
    args = ap.parse_args()

    print(f'connecting to {args.connection} ...')
    m = mavutil.mavlink_connection(args.connection)
    m.wait_heartbeat()
    print(f'heartbeat: system {m.target_system} component {m.target_component}')

    print('waiting for EKF / GPS lock ...')
    if not wait_ekf_ready(m, args.ekf_timeout):
        print('EKF never became ready', file=sys.stderr)
        return 1
    print('EKF ready')

    # set_mode() is fire-and-forget in pymavlink, and GUIDED is REJECTED for a
    # while after EKF_STATUS_REPORT first goes green -- the mode needs a
    # position estimate the EKF flags claim before it is really usable. Arming
    # then happens in STABILIZE, MAV_CMD_NAV_TAKEOFF is ACKed but ignored
    # (STABILIZE has no takeoff), and the vehicle sits on the ground spinning
    # while the climb loop times out. So confirm the mode actually took.
    if not set_mode_confirmed(m, 'GUIDED', args.mode_timeout):
        print('could not enter GUIDED (needs a position estimate)',
              file=sys.stderr)
        return 1
    print('mode GUIDED confirmed')

    # Pre-arm checks can still be settling right after EKF flags go green, so
    # retry rather than bailing on the first refusal.
    print('arming ...')
    armed = False
    for attempt in range(1, 11):
        m.arducopter_arm()
        if wait_armed(m, 5):
            armed = True
            break
        print(f'  ... arm attempt {attempt} not accepted yet')
    if not armed:
        print('failed to arm (check pre-arm messages in the SITL pane)',
              file=sys.stderr)
        return 1
    print('armed')

    def send_takeoff():
        print(f'takeoff to {args.alt} m')
        m.mav.command_long_send(
            m.target_system, m.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
            0, 0, 0, 0, 0, 0, args.alt)

    if not set_mode_confirmed(m, 'GUIDED', 15):
        print('fell out of GUIDED before takeoff', file=sys.stderr)
        return 1
    send_takeoff()

    # Settle for 95% of the target so we don't sit here on the last few cm.
    target = args.alt * 0.95
    deadline = time.time() + args.climb_timeout
    # ArduPilot ACKs the takeoff and then silently refuses to climb if the EKF
    # is still aligning -- which happens on a freshly-wiped eeprom, where the
    # EKF_STATUS_REPORT flags go green a good while before the vehicle is
    # actually ready ("in-flight yaw alignment complete" arrives later). The
    # symptom is a long run of "alt 0.0" and then a timeout. Re-issuing the
    # command once the vehicle is genuinely ready gets it moving.
    stalled_since = time.time()
    retries_left = args.takeoff_retries
    while time.time() < deadline:
        msg = m.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5)
        if not msg:
            continue
        alt = msg.relative_alt / 1000.0
        print(f'  alt {alt:5.1f} m')
        if alt >= target:
            print(f'reached {alt:.1f} m -- hovering, ready for a goal')
            return 0

        if alt > 0.5:
            stalled_since = time.time()      # climbing; reset the stall timer
        elif time.time() - stalled_since > args.stall_timeout and retries_left:
            retries_left -= 1
            print(f'  still on the ground after {args.stall_timeout:.0f}s '
                  f're-sending takeoff ({retries_left} retries left)')
            send_takeoff()
            stalled_since = time.time()

    print('did not reach takeoff altitude in time', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
