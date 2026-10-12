"""Hold ``/clock`` at the bag's last time until DAAAM has drained its queues.

Hydra's node (``exit_after_clock: true``) exits as soon as no ``/clock``
publisher is left, and the launch's ``on_exit: shutdown`` then stops the DAAAM
node, which saves whatever has been grounded so far. Upstream runs the bag in
real time on long CODa sequences, where the last query interval is a small
tail; on a one-minute indoor trajectory it is a large share of the map. So the
adapter starts this node *before* the bag (its publisher keeps Hydra's
``haveClock`` true across the hand-over), and once the bag has finished it
publishes the bag's last time, unchanged, like a paused ``ros2 bag play``.
When the adapter writes the stop file, this node exits; Hydra then sees no
clock and shuts the launch down exactly as it does after a bag upstream.

    python -m ragmap_adapter.clock_keeper --time-ns N --hold-file F --stop-file S
"""
from __future__ import annotations

import argparse
import os
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-ns", type=int, required=True)
    parser.add_argument("--hold-file", required=True)
    parser.add_argument("--stop-file", required=True)
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from rosgraph_msgs.msg import Clock

    rclpy.init()
    node = Node("ragmap_clock_keeper")
    publisher = node.create_publisher(Clock, "/clock", 10)
    message = Clock()
    message.clock.sec = args.time_ns // 1_000_000_000
    message.clock.nanosec = args.time_ns % 1_000_000_000
    try:
        while not os.path.exists(args.stop_file):
            if os.path.exists(args.hold_file):
                publisher.publish(message)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
