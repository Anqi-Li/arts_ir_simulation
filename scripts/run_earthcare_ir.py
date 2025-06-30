# %%
import os
import sys
import numpy as np
import subprocess


# %% create a loop to run ../src/ir_earthcare.py for several input arguments
def run_earthcare_ir(orbit, i, j):
    """
    Run the EarthCARE IR processing script with the specified input and output files.

    Parameters:
    i (int): index of habit
    j (int): index of psd.
    """

    command = [
        # sys.executable,  # Use the current Python interpreter
        "python",  # Use 'python' to run the script
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/ir_earthcare.py"),
        str(i),
        str(j),
        orbit,
    ]

    subprocess.run(command, check=True)


# %% Loop through the habit and psd indices
orbit = "01162E"  # Default orbit frame
for i in range(2):
    for j in range(3):
        print(f"Running EarthCARE IR for orbit {orbit}, habit {i} and psd {j}")
        run_earthcare_ir(orbit, i, j)
        print(f"Completed EarthCARE IR for orbit {orbit}, habit {i} and psd {j}")
print("All EarthCARE IR processing completed.")
