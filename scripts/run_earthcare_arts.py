# %%
import os
import subprocess
from datetime import datetime
import glob
from earthcare_ir import habit_std_list, psd_list

log_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "log",
)
os.makedirs(log_dir, exist_ok=True)

# Create a new log file with timestamp
log_path = os.path.join(log_dir, f"run_earthcare_aws_{datetime.now():%Y%m%d_%H%M%S}.log")

# Keep only the 5 most recent log files
log_files = sorted(glob.glob(os.path.join(log_dir, "run_earthcare_aws_*.log")))
while len(log_files) > 4:  # will become 5 after this run
    os.remove(log_files[0])
    log_files = sorted(glob.glob(os.path.join(log_dir, "run_earthcare_aws_*.log")))


def run_arts(orbit, i, j):
    command = [
        "python",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/earthcare_aws.py"),
        str(i),
        str(j),
        orbit,
    ]
    # Only capture stderr, let stdout go to terminal
    result = subprocess.run(command, check=True, stderr=subprocess.PIPE, text=True)
    # Do not print stdout/stderr for successful runs


orbit_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/earthcare/arts_input_data")
orbits = [o[-9:-3] for o in os.listdir(orbit_dir)]

with open(log_path, "w") as logfile:
    for orbit in orbits:
        failed = False
        for i in range(len(habit_std_list)):
            for j in range(len(psd_list)):
                try:
                    run_arts(orbit, i, j)
                    print(f"Success for orbit {orbit}, habit {i}, psd {j}")
                except subprocess.CalledProcessError as e:
                    logfile.write(f"Failed for orbit {orbit}, habit {i}, psd {j}: {e}\n")
                    logfile.write("Subprocess stderr:\n")
                    logfile.write(f"{e.stderr}\n")
                    failed = True
                    break  # break out of the j loop
            if failed:
                break  # break out of the i loop

print("All EarthCARE IR processing completed.")

# %%
