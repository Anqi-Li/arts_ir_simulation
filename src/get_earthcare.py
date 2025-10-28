# %%
from pathlib import Path
import numpy as np
from ectools import ecio
import data_paths as dp
import os
from scipy.interpolate import NearestNDInterpolator

# %% fetch all arts output files
# TODO: use the new function in ecio to merge C-FMR and M-RGR data
def load_and_merge_input_data(frame_code, product_baseline="BA", save_results=False, cfmr_vars=None, get_xmet_temperature=False):
    """Load and merge C-FMR and M-RGR data for a given orbit frame code."""
    # load C-FMR data
    ds_cfmr = ecio.load_CFMR(
        srcpath=dp.CFMR,
        product_baseline=product_baseline,
        frame=frame_code[-1],
        orbit=frame_code[:-1],
        nested_directory_structure=True,
    )
    ds_cfmr.close()
    print("C-FMR loading done.")

    if get_xmet_temperature:
        ds_xmet = ecio.load_XMET(
            srcpath=dp.XMET,
            product_baseline="AA",
            frame=frame_code[-1],
            orbit=frame_code[:-1],
            nested_directory_structure=True,
        )
        ds_xmet.close()
        print("X-MET loading done.")
        # merge XMET data into ds_cfmr
        ds_cfmr = ecio.get_XMET(
            ds_xmet,
            ds_cfmr,
            XMET_1D_variables=[],
            XMET_2D_variables=["temperature"],
        )
        print("Merging XMET data into input dataset (C-FMR) done.")

    ds_cfmr = ds_cfmr.set_coords(["longitude", "latitude", "time", "height", "surface_elevation"])
    if cfmr_vars is not None:
        if get_xmet_temperature and "temperature" not in cfmr_vars:
            cfmr_vars.append("temperature")
        ds_cfmr = ds_cfmr[cfmr_vars]

    # load M-RGR data
    ds_mrgr = ecio.load_MRGR(
        srcpath=dp.MRGR,
        product_baseline=product_baseline,
        frame=frame_code[-1],
        orbit=frame_code[:-1],
        nested_directory_structure=True,
    )
    ds_mrgr.close()
    print("M-RGR loading done.")
    # pick the nearest M-RGR pixel for each CPR ray
    flatten_hcoords_mrgr = (
        ds_mrgr.reset_coords(["longitude", "latitude"])[["longitude", "latitude"]]
        .stack({"horizontal_grid": ["along_track", "across_track"]})
        .to_array()
    )
    NearestIndex = NearestNDInterpolator(
        flatten_hcoords_mrgr.data.T,
        np.arange(len(flatten_hcoords_mrgr["horizontal_grid"])),
    )
    nearest_indices_on_flatten_hcoords_mrgr = NearestIndex(np.array([ds_cfmr["longitude"], ds_cfmr["latitude"]]).T).astype(int)
    ds_mrgr_TIR_select = (
        ds_mrgr[["TIR1", "TIR2", "TIR3"]]
        .stack({"horizontal_grid": ["along_track", "across_track"]})
        .isel({"horizontal_grid": nearest_indices_on_flatten_hcoords_mrgr})
    )
    print("Selecting nearest M-RGR TIR2 pixel for each CPR ray done.")

    # merge datasets
    ds_ec = ds_cfmr.assign({"msi": (("band", "along_track"), ds_mrgr_TIR_select.to_array().data)})
    ds_ec["msi"].attrs = ds_mrgr.TIR1.attrs
    ds_ec = ds_ec.assign_coords({"band": ["TIR1", "TIR2", "TIR3"]})
    ds_ec.attrs.update(
        {
            "CPR source": ds_cfmr.encoding["source"].split("/")[-1],
            "MSI source": ds_mrgr.encoding["source"].split("/")[-1],
        }
    )
    if get_xmet_temperature:
        ds_ec.attrs.update(
            {
                "XMET source": ds_xmet.encoding["source"].split("/")[-1],
            }
        )
    print("Merging C-FMR and M-RGR done.")

    if save_results:
        save_path = os.path.join(
            dp.MRGR_TIR_aligned,
            f"CPR_MSI_merged_{frame_code}.nc",
        )
        ds_ec.to_netcdf(save_path)
        print(f"Saved merged dataset to {save_path}")
    # ds = ds_ec.set_coords(ds_arts.coords).merge(ds_arts)
    # print("Merging ARTS result with dataset C-FMR and M-RGR done.")
    return ds_ec


# %%
if __name__ == "__main__":
    # %% orbit frames based on arts output files
    # from earthcare_ir import Habit, PSD

    # def get_arts_output_files(orbit_frame, habit, psd):
    #     import data_paths as dp
    #     import os
    #     import glob

    #     path_pattern = os.path.join(dp.arts_output_TIR2, f"arts_TIR2_{orbit_frame}_{habit}_{psd}.nc")
    #     return glob.glob(path_pattern)

    # arts_file_paths = get_arts_output_files("*", Habit.Column, PSD.D14)
    # orbit_frames = [f.split("/")[-1].split("_")[2] for f in arts_file_paths]

    # %% get common orbit frames in August 2025
    orbit_frames = dp.get_common_orbit_frame_list(year="2025", month="08", day="*")

    # %%
    for orbit_frame in orbit_frames[:2]:
        print(f"Processing orbit frame: {orbit_frame}")
        save_path = os.path.join(
            dp.MRGR_TIR_aligned,
            f"CPR_MSI_merged_{orbit_frame}.nc",
        )
        if Path(save_path).exists():
            print(f"File {save_path} already exists, skipping.")
            # continue
        ds = load_and_merge_input_data(
            frame_code=orbit_frame,
            save_results=False,
            cfmr_vars=["reflectivity_corrected"],
            get_xmet_temperature=True,
        )
