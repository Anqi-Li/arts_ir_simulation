# %%
import os
import subprocess
from ectools import ecio
import data_paths as dp
from datetime import datetime
import argparse
import tempfile

# %%
# Parse arguments first
parser = argparse.ArgumentParser(description="Run cap2tb_aws_ir.py for multiple orbits")
parser.add_argument(
    "--max-processes",
    type=int,
    default=8,
    help="Maximum number of parallel processes (default: 8)",
)
args = parser.parse_args()

# %%
filelist = ecio.get_filelist(
    basedir=dp.ACMCAP,
    product_baseline="BA",
    nested_directory_structure=True,
    start_range_str="20250701T000000Z",
    end_range_str="20250731T235959Z",
)
list_orbit_frame = [f.split("/")[-1].split("_")[-1].split(".")[0] for f in filelist]

# %%
# Configuration
MAX_PROCESSES = args.max_processes
skip_profile = 50

ice_habit_ir = [
    "8-ColumnAggregate-ModeratelyRough",
    "Plate-ModeratelyRough",
    "SolidBulletRosette-ModeratelyRough",
    "Droxtal-ModeratelyRough",
]
ice_habit_aws = [
    "8-ColumnAggregate",
    "LargePlateAggregate",
    "6-BulletRosette",
    "LargeColumnAggregate",
]

log_file_path = f"/home/anqil/arts_ir_simulation/data/log/cap2tb_aws_ir_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
failed_file_path = (
    "/home/anqil/arts_ir_simulation/data/log/cap2tb_aws_ir_failed_orbit_frames2.txt"
)

# Load previously failed orbit_frames to skip
skip_failed = set()
if os.path.exists(failed_file_path):
    with open(failed_file_path, "r") as f:
        skip_failed = set(line.strip() for line in f if line.strip())
    print(f"Loaded {len(skip_failed)} previously failed orbit_frames to skip.")

failed_orbit_frame = []


def build_cmd(orbit_frame):
    """Build command for a given orbit_frame."""
    orbit = orbit_frame[:-1]
    frame = orbit_frame[-1]
    output = f"/home/anqil/arts_ir_simulation/data/temp_files/new_dmean/cap2tb_aws_ir_{orbit}_{frame}_skip_{skip_profile}.nc"

    cmd = [
        "python3",
        "/home/anqil/arts_ir_simulation/src/cap2tb_aws_ir.py",
        "--orbit",
        f"{orbit}",
        "--frame",
        f"{frame}",
        "--skip",
        f"{skip_profile}",
        "--ice-habit-ir",
        ",".join(ice_habit_ir),
        "--ice-habit-aws",
        ",".join(ice_habit_aws),
        "--output",
        f"{output}",
    ]
    return cmd, output


def save_failed_list():
    """Save failed orbit_frames to file (append new failures)."""
    all_failed = skip_failed.union(set(failed_orbit_frame))
    with open(failed_file_path, "w") as f:
        for of in sorted(all_failed):
            f.write(f"{of}\n")
    print(f"Saved {len(all_failed)} failed orbit_frames to {failed_file_path}")


# %%
# Main execution with process pool
print(
    f"Starting job runner with max {MAX_PROCESSES} processes for {len(list_orbit_frame)} orbits..."
)
print(f"Log file: {log_file_path}")
print(f"Failed list file: {failed_file_path}\n")

with open(log_file_path, "w") as log_file:
    log_file.write(f"Started at {datetime.now()}\n")
    log_file.write(f"Max parallel processes: {MAX_PROCESSES}\n")
    log_file.write(f"Total orbits: {len(list_orbit_frame)}\n")
    log_file.write(f"Skipping {len(skip_failed)} previously failed orbits\n\n")

# Track running processes: {process: (orbit_frame, stderr_file_path)}
running = {}
completed = 0
skipped = 0
skipped_failed = 0
total = len(list_orbit_frame)
orbit_iter = iter(list_orbit_frame)

try:
    while completed + skipped + skipped_failed < total:
        # Fill up to MAX_PROCESSES
        while len(running) < MAX_PROCESSES:
            try:
                orbit_frame = next(orbit_iter)
            except StopIteration:
                break  # No more orbits to submit

            cmd, output = build_cmd(orbit_frame)

            # Skip if previously failed
            if orbit_frame in skip_failed:
                print(f"[{orbit_frame}] Previously failed, skipping.")
                skipped_failed += 1
                continue

            # Skip if output exists
            if os.path.exists(output):
                print(f"[{orbit_frame}] Output exists, skipping.")
                skipped += 1
                continue

            # Create a temp file for stderr to avoid pipe buffer deadlock
            stderr_file = tempfile.NamedTemporaryFile(
                mode="w+", delete=False, suffix=f"_{orbit_frame}.err"
            )

            # Start new process - suppress stdout, write stderr to temp file
            print(
                f"[{orbit_frame}] Starting... (running: {len(running) + 1}/{MAX_PROCESSES})"
            )
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,  # Suppress normal output
                stderr=stderr_file,  # Write errors to temp file (no buffer limit)
            )
            running[proc] = (orbit_frame, stderr_file.name)
            stderr_file.close()  # Close our handle, subprocess keeps writing

        if not running:
            break  # No processes running and no more to start

        # Wait for at least one process to finish
        import time

        finished_any = False
        while not finished_any and running:
            for proc in list(running.keys()):
                retcode = proc.poll()  # Non-blocking check
                if retcode is not None:
                    orbit_frame, stderr_path = running.pop(proc)
                    completed += 1
                    finished_any = True

                    if retcode == 0:
                        print(
                            f"[{orbit_frame}] ✓ Completed ({completed}/{total - skipped - skipped_failed}) | Running: {len(running)}"
                        )
                        # Clean up temp file on success
                        if os.path.exists(stderr_path):
                            os.remove(stderr_path)
                    else:
                        failed_orbit_frame.append(orbit_frame)
                        print(
                            f"[{orbit_frame}] ❌ Failed ({completed}/{total - skipped - skipped_failed}) | Running: {len(running)}"
                        )
                        # Log error details to main log file
                        with open(log_file_path, "a") as log_file:
                            log_file.write(f"\n{'='*70}\n")
                            log_file.write(
                                f"ERROR in {orbit_frame} at {datetime.now()}\n"
                            )
                            log_file.write(f"Return code: {retcode}\n")
                            log_file.write(f"{'='*70}\n")
                            # Read stderr from temp file
                            if os.path.exists(stderr_path):
                                with open(stderr_path, "r") as ef:
                                    stderr_output = ef.read()
                                    if stderr_output:
                                        log_file.write(stderr_output)
                                os.remove(stderr_path)
                            log_file.write(f"\n{'='*70}\n\n")

            if not finished_any and running:
                time.sleep(0.5)

except KeyboardInterrupt:
    print("\n\nInterrupted! Saving failed list before exit...")
    # Terminate running processes and clean up temp files
    for proc, (orbit_frame, stderr_path) in running.items():
        proc.terminate()
        if os.path.exists(stderr_path):
            os.remove(stderr_path)
    save_failed_list()
    raise

# Save failed list at the end
save_failed_list()

# Final summary
with open(log_file_path, "a") as log_file:
    log_file.write(f"\n{'='*70}\n")
    log_file.write(f"Completed at {datetime.now()}\n")
    log_file.write(
        f"Processed: {completed}, Skipped (exists): {skipped}, Skipped (failed): {skipped_failed}, New failures: {len(failed_orbit_frame)}\n"
    )
    log_file.write(f"New failed list: {failed_orbit_frame}\n")
    log_file.write(f"{'='*70}\n")

print(f"\n{'='*70}")
print(
    f"Processed: {completed}, Skipped (exists): {skipped}, Skipped (prev failed): {skipped_failed}"
)
print(f"{'='*70}")
