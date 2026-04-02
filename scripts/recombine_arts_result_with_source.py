# %%
import re
from ectools import ecio
from contextlib import redirect_stdout
import os
import data_paths as dp
import xarray as xr


# %%
def combine_acmcap(ds):
    filename = ds.attrs.get("ACMCAP_source")
    # print(filename)
    pattern = r"ECA_EX(?P<baseline>\w+)_\w+_\w+_\w+_\w+_\w+_(?P<orbit>\d{5})(?P<frame>[A-Z])\.h5"
    match = re.match(pattern, filename)
    if match:
        orbit = match.group("orbit")  # '06203'
        frame = match.group("frame")  # 'D'
        baseline = match.group("baseline")  # 'BA'
        with redirect_stdout(open(os.devnull, "w")):
            ds_acmcap = ecio.load_ACMCAP(
                srcpath=dp.ACMCAP,
                product_baseline=baseline,
                orbit=orbit,
                frame=frame,
                nested_directory_structure=True,
            )
            ds_acmcap.close()  # Close the original dataset to free up resources
        ds = ds.merge(
            ds_acmcap[
                [
                    "height",
                    "elevation",
                    "ice_water_path",
                    "ice_water_content",
                    "ice_riming_factor",
                    "liquid_water_content",
                    "rain_water_content",
                    "MSI_longwave_brightness_temperature",
                    "MSI_longwave_brightness_temperature_forward",

                ]
            ],
            join="inner",
        )
    return ds


def combine_xmet(ds):
    filename = ds.attrs.get("XMET_source")
    pattern = r"ECA_EX(?P<baseline>\w+)_\w+_\w+_\w+_\w+_\w+_(?P<orbit>\d{5})(?P<frame>[A-Z])\.h5"
    match = re.match(pattern, filename)
    if match:
        orbit = match.group("orbit")  # '06203'
        frame = match.group("frame")  # 'D'
        # baseline = match.group('baseline') # 'BA'
        with redirect_stdout(open(os.devnull, "w")):
            ds_xmet = ecio.load_XMET(
                srcpath=dp.XMET,
                orbit=orbit,
                frame=frame,
                # baseline=baseline,
                nested_directory_structure=True,
            )
            ds_xmet.close()  # Close the original dataset to free up resources
            ds = ecio.get_XMET(
                ds_xmet,
                ds,
                XMET_2D_variables=[
                    "temperature",
                    "pressure",
                    "specific_humidity",
                ],
                XMET_1D_variables=['sea_ice_cover'],
            )
    return ds

#%%
if __name__ == "__main__":
    # %%
    result_path = "/home/anqil/arts_ir_simulation/data/temp_files/new_dmean"
    matching_files = [f for f in os.listdir(result_path) if f.endswith("_skip_50.nc")]
    print(f"{len(matching_files)} files match the pattern")

    # %%
    for i in range(len(matching_files)):
        print(f"{i+1}/{len(matching_files)}")
        orbit = matching_files[i].split("_")[3]  # '06203'
        frame = matching_files[i].split("_")[4]  # 'D'
        output_path = "/home/anqil/arts_ir_simulation/data/temp_files/july"
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"cap2tb_aws_ir_{orbit}_{frame}_skip_50.nc")
        
        # if output file already exists, skip saving to avoid overwriting
        if os.path.exists(output_file):
            print(f"{output_file} already exists. Skipping saving.")
        else:
            ds = xr.open_dataset(os.path.join(result_path, matching_files[i]))
            # print(ds.attrs.get("ACMCAP_source"))
            # print(ds.attrs.get("XMET_source"))
            ds = combine_acmcap(ds)
            ds = combine_xmet(ds)

            # save the combined dataset to a new NetCDF file
            ds.to_netcdf(output_file)
            print(f"Combined dataset saved to {output_file}")

