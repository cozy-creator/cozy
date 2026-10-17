#!/usr/bin/env bash
# shaped-link.sh [--port P] [--kbit K] [--delay-ms D] | --off
# Shapes loopback traffic to and from one TCP port like a slow, far link, on lo at MTU 1500: each direction gets its
# own netem, delaying packets D ms (the round trip is 2D) at K kbit/s and queueing one round trip of packets beyond
# those in flight (netem's limit counts delayed packets too).
# Prints COZY_TEST_SHAPED_BPS for Calcifer's shaped tests (crates/calcifer/tests/shaped.rs, run with --ignored), which
# start their machine on the port. Defaults: the owner's uplink, 2.6 Mbit/s and 400 ms. Needs sudo.
#   export $(scripts/shaped-link.sh --kbit 8000 | tail -n1) && cargo test -p calcifer --test shaped -- --ignored
set -euo pipefail
port=47123 kbit=2600 delay=200 off=
while [ $# -gt 0 ]; do
  case $1 in
    --port) port=$2; shift ;;
    --kbit) kbit=$2; shift ;;
    --delay-ms) delay=$2; shift ;;
    --off) off=1 ;;
    *) sed -n '2,8s/^# \{0,1\}//p' "$0" >&2; exit 2 ;;
  esac
  shift
done
sudo tc qdisc del dev lo root 2>/dev/null || true
if [ -n "$off" ]; then
  sudo ip link set dev lo mtu 65536
  exit 0
fi
bps=$((kbit * 125))
limit=$(((bps * delay * 3 + 1514000 - 1) / 1514000))
sudo ip link set dev lo mtu 1500
sudo ethtool -K lo gso off tso off gro off 2>/dev/null || true
sudo tc qdisc add dev lo root handle 1: prio bands 4 priomap 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
band=3
for side in dport sport; do
  sudo tc qdisc add dev lo parent 1:$band handle ${band}0: netem delay "${delay}ms" rate "${kbit}kbit" limit "$limit"
  sudo tc filter add dev lo parent 1: protocol ip prio 1 u32 match ip "$side" "$port" 0xffff flowid 1:$band
  band=4
done
tc qdisc show dev lo >&2
echo "COZY_TEST_SHAPED_BPS=$bps"
