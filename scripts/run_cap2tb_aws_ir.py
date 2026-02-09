# %%
import os
import subprocess
from ectools import ecio
import data_paths as dp
from datetime import datetime

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
failed_orbit_frame = []
log_file_path = f"/home/anqil/arts_ir_simulation/data/log/cap2tb_aws_ir_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_number = len(list_orbit_frame)
with open(log_file_path, "w") as log_file:
    for orbit_frame in list_orbit_frame[:file_number//2][::-1]:
        try:
            orbit = orbit_frame[:-1]
            frame = orbit_frame[-1]
            ice_habit_ir = [
                "8-ColumnAggregate-ModeratelyRough",
                "Plate-ModeratelyRough",
                "SolidBulletRosette-ModeratelyRough",
                "Droxtal-ModeratelyRough",
            ]
            ice_habit_aws = [
                "LargePlateAggregate",
                "LargeColumnAggregate",
                "6-BulletRosette",
                "8-ColumnAggregate",
            ]
            # pr0 = 0
            # prn = 10
            skip = 100
            output = f"/home/anqil/arts_ir_simulation/data/temp_files/cap2tb_aws_ir_{orbit}_{frame}_skip_{skip}.nc"
            if os.path.exists(output):
                print(
                    f"Output file already exists. Skipping orbit_frame {orbit_frame}."
                )
                continue

            print(f"Processing {orbit_frame}...")

            cmd = [
                "python3",
                "/home/anqil/arts_ir_simulation/src/cap2tb_aws_ir.py",
                "--orbit",
                f"{orbit}",
                "--frame",
                f"{frame}",
                # "--pr0",
                # f"{pr0}",
                # "--prn",
                # f"{prn}",
                "--skip",
                f"{skip}",
                "--ice-habit-ir",
                f'{",".join(ice_habit_ir)}',
                "--ice-habit-aws",
                f'{",".join(ice_habit_aws)}',
                "--output",
                f"{output}",
            ]

            # Create a temporary file for error capture
            error_log_temp = f"/tmp/arts_error_{orbit_frame}.log"

            # Use shell to tee stderr - show in terminal AND capture to temp file
            cmd_str = " ".join([f'"{arg}"' for arg in cmd])
            shell_cmd = f"{cmd_str} 2> >(tee {error_log_temp} >&2)"

            result = subprocess.run(shell_cmd, shell=True, executable="/bin/bash")

            if result.returncode != 0:
                failed_orbit_frame.append(orbit_frame)
                log_file.write(f"\n{'='*70}\n")
                log_file.write(f"ERROR in {orbit_frame} at {datetime.now()}\n")
                log_file.write(f"Return code: {result.returncode}\n")
                log_file.write(f"{'='*70}\n")

                # Read error from temp file and write to log
                try:
                    with open(error_log_temp, "r") as ef:
                        error_content = ef.read()
                        log_file.write(error_content)
                    os.remove(error_log_temp)  # Clean up
                except FileNotFoundError:
                    log_file.write("No error details captured\n")

                log_file.write(f"\n{'='*70}\n\n")
                log_file.flush()
                print(f"  ❌ Failed")
            else:
                # Clean up temp file if no error
                if os.path.exists(error_log_temp):
                    os.remove(error_log_temp)
                print(f"  ✓ Completed successfully")

        except Exception as e:
            print(f"  ❌ Exception: {e}")
            log_file.write(f"\n{'='*70}\n")
            log_file.write(f"EXCEPTION in {orbit_frame} at {datetime.now()}\n")
            log_file.write(f"Exception: {e}\n")
            log_file.write(f"{'='*70}\n\n")
            log_file.flush()
            failed_orbit_frame.append(orbit_frame)
            continue

    log_file.write(
        f"\n{'='*70}\nCompleted at {datetime.now()}\nFailed: {failed_orbit_frame}\n{'='*70}\n"
    )
    print(
        f"\nCompleted. Failed orbit_frames ({len(failed_orbit_frame)}): {failed_orbit_frame}"
    )
