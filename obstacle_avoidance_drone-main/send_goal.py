#!/usr/bin/env python3
"""
Send a hardcoded lat/lon goal to goal_socket_bridge over UDP.

Plain Python — no ROS, no colcon build. Run it on the OBC (or anywhere that can
reach the OBC) while goal_socket_bridge.py and Nav2 are up:

    python3 send_goal.py                        # uses GOAL_LAT / GOAL_LON below
    python3 send_goal.py -35.36270 149.16523    # override on the command line
    python3 send_goal.py --cancel               # cancel the active goal

The bridge converts lat/lon -> map frame using the drone's current fix, then
drives Nav2 via NavigateToPose. Status replies (converted / accepted /
feedback / reached / aborted) are printed here until the goal finishes.
Ctrl+C sends a cancel before exiting.
"""

import argparse
import json
import socket
import sys

# ---- HARDCODED GOAL ----
GOAL_LAT = -35.36270
GOAL_LON = 149.16523
GOAL_YAW = 0.0

# goal_socket_bridge's bind_host/bind_port. Localhost when both run on the OBC.
BRIDGE_HOST = '127.0.0.1'
BRIDGE_PORT = 9200

# Statuses that end the run.
TERMINAL = {'reached', 'aborted', 'canceled', 'rejected'}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('lat', nargs='?', type=float, default=GOAL_LAT)
    ap.add_argument('lon', nargs='?', type=float, default=GOAL_LON)
    ap.add_argument('--yaw', type=float, default=GOAL_YAW,
                    help='goal heading in radians (ENU), default %(default)s')
    ap.add_argument('--host', default=BRIDGE_HOST)
    ap.add_argument('--port', type=int, default=BRIDGE_PORT)
    ap.add_argument('--cancel', action='store_true',
                    help='cancel the active goal instead of sending one')
    ap.add_argument('--timeout', type=float, default=300.0,
                    help='give up waiting for replies after N seconds')
    args = ap.parse_args()

    addr = (args.host, args.port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(args.timeout)

    if args.cancel:
        msg = {'cmd': 'cancel'}
    else:
        msg = {'cmd': 'goal', 'lat': args.lat, 'lon': args.lon, 'yaw': args.yaw}
        print(f'-> goal lat={args.lat:.7f} lon={args.lon:.7f} yaw={args.yaw:.2f} '
              f'to {args.host}:{args.port}')

    sock.sendto(json.dumps(msg).encode(), addr)

    try:
        while True:
            try:
                data, _ = sock.recvfrom(2048)
            except socket.timeout:
                print('no reply — is goal_socket_bridge running on '
                      f'{args.host}:{args.port}?', file=sys.stderr)
                return 1

            try:
                reply = json.loads(data.decode())
            except (ValueError, UnicodeDecodeError):
                print(f'<- unparseable reply: {data!r}', file=sys.stderr)
                continue

            status = reply.get('status', '?')
            print(f'<- {status}: {reply.get("msg", "")}')

            if status in TERMINAL:
                return 0 if status == 'reached' else 1
            if args.cancel and status == 'canceling':
                return 0
    except KeyboardInterrupt:
        print('\ninterrupted — sending cancel')
        sock.sendto(json.dumps({'cmd': 'cancel'}).encode(), addr)
        return 130


if __name__ == '__main__':
    sys.exit(main())
