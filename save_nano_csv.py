import argparse
import csv
import os
import serial


def main():
    parser = argparse.ArgumentParser(description="Save Nano CNN CSV windows to a file")
    parser.add_argument("port", help="Serial port, for example COM7")
    parser.add_argument("--output", default="imu_predictions.csv")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    in_csv = False
    header_written = os.path.exists(args.output) and os.path.getsize(args.output) > 0

    with serial.Serial(args.port, args.baud, timeout=1) as device, open(
        args.output, "a", newline=""
    ) as output_file:
        writer = None
        print("Logging to " + args.output + ". Press Ctrl+C to stop.")

        while True:
            line = device.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line == "CSV_BEGIN":
                in_csv = True
                writer = None
                continue
            if line == "CSV_END":
                in_csv = False
                continue
            if not in_csv:
                continue

            row = next(csv.reader([line]))
            if writer is None:
                writer = csv.writer(output_file)
                if not header_written:
                    writer.writerow(row)
                    header_written = True
            else:
                writer.writerow(row)
            output_file.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nLogging stopped.")
