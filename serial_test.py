#!/usr/bin/env python3
"""
Quick serial hardware test for the Moodify ESP32 actuator.

- Opens the given serial port and sends:
  1) TEST (full-strip test)
  2) BRIGHT 64 (set brightness lower)
  3) CHASE (pixel chase diagnostic)

Usage:
  python3 serial_test.py --port /dev/cu.SLAB_USBtoUART --baud 115200

Notes:
- Make sure Arduino Serial Monitor is closed before running.
- If you see a 'Resource busy' error, another process is holding the port.
"""
import argparse
import time

try:
    import serial  # type: ignore
except Exception as e:
    raise SystemExit("pyserial not installed. pip install pyserial")


def send(ser, cmd: str):
    if not cmd.endswith("\n"):
        cmd += "\n"
    ser.write(cmd.encode("utf-8"))
    ser.flush()
    print(f"-> {cmd.strip()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--wait", type=float, default=1.8, help="Wait after open for ESP32 reset (s)")
    args = p.parse_args()

    with serial.Serial(port=args.port, baudrate=args.baud, timeout=0.1) as ser:
        # Give ESP32 time to reset on open
        time.sleep(max(0.0, args.wait))
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        # Send a short test sequence
        send(ser, "TEST")
        time.sleep(0.5)
        send(ser, "BRIGHT 64")
        time.sleep(0.5)
        send(ser, "CHASE")
        print("Sent TEST/BRIGHT/CHASE. Observe LEDs.")


if __name__ == "__main__":
    main()
