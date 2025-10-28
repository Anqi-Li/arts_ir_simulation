# %%
import os
from cycler import cycler
from earthcare_ir import get_cloud_top_T
from matplotlib.colors import LogNorm
from pyarts.workspace import Workspace
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from earthcare_ir import habit_std_list, psd_list
import xarray as xr
import glob
import xgboost as xgb
import data_paths as dp
import pandas as pd
from ectools import ecio


# %% analysis on output data
def load_and_merge_ec_data(
    ds_arts,
    merge_kwargs=dict(join="inner", overwrite_vars=["longitude", "latitude", "time"]),
    verbose=False,
):
    orbit_frame = ds_arts.arts.attrs["CPR source"].split("_")[-1].split(".")[0]
    ds_ec = xr.open_dataset(
        os.path.join(dp.MRGR_TIR_aligned, f"CPR_MSI_merged_{orbit_frame}.nc"),
        chunks="auto",
    )

    ds = ds_ec.merge(ds_arts, **merge_kwargs)

    if verbose:
        print("Merging EarthCare data and ARTS results done.")

    return ds


def load_arts_results(habit="*", psd="*", orbit_frame="*", verbose=False):
    file_path_str = os.path.join(
        dp.arts_output_TIR2, f"arts_TIR2_{orbit_frame}_{habit}_{psd}.nc"
    )
    ds_arts = xr.open_mfdataset(
        file_path_str,
        combine="nested",
        parallel=True,
        chunks="auto",
        preprocess=lambda ds: ds.assign_coords(
            {"psd": (ds.arts.attrs["PSD"]), "habit": (ds.arts.attrs["habit"])}
        ).expand_dims(["psd", "habit"]),
    )
    ds_arts.close()

    if verbose:
        print("Loading existing ARTS result done.")

    return ds_arts


def load_and_merge_acmcap_data(ds_arts, keep_vars=None, orbit_frame=None, **kwargs):
    if orbit_frame is None:
        try:
            orbit_frame = ds_arts.orbit_frame.item()
        except Exception as e:
            raise ValueError(
                "orbit_frame must be provided if not found in ds_arts encoding."
            ) from e
    ds_acmcap = (
        ecio.load_ACMCAP(
            srcpath=dp.ACMCAP,
            product_baseline="BA",
            orbit=orbit_frame[:-1],
            frame=orbit_frame[-1],
            nested_directory_structure=True,
            **kwargs,
        )
        .swap_dims(along_track="time")
        .reset_coords()
    ).chunk("auto")
    if keep_vars is not None:
        ds_acmcap = ds_acmcap[keep_vars]
    ds_acmcap.close()

    ds_acmcap = ds_acmcap.rename({"MSI_longwave_channel": "band"}).assign(
        {"band": ["TIR1", "TIR2", "TIR3"]}
    )

    ds_acmcap_re = ds_acmcap.reindex_like(
        ds_arts.swap_dims(along_track="time"),
        method="nearest",
        tolerance=pd.Timedelta("1s"),
    )
    ds_merged = ds_arts.swap_dims(along_track="time").merge(
        ds_acmcap_re, overwrite_vars=["along_track", "latitude", "longitude"]
    )
    ds_merged.attrs["ACMCAP source"] = ds_acmcap.encoding.get(
        "source", "unknown"
    ).split("/")[-1]
    return ds_merged.swap_dims(time="along_track")


def get_bias(ds, var_name1, var_name2):
    return (ds[var_name1] - ds[var_name2]).assign_attrs(
        {
            "long_name": f"{var_name1} - {var_name2} Bias",
            "units": ds[var_name1].attrs.get("units"),
        }
    )


def get_var_at_max_height_by_dbz(
    ds, threshold_dbz, var_name, dbz_name="reflectivity_corrected"
):
    cond = ds[dbz_name] > threshold_dbz

    # ensure cond dims in order (CPR_height, along_track)
    if cond.dims != ("CPR_height", "along_track"):
        cond = cond.transpose("CPR_height", "along_track")

    # which profiles have at least one True
    has_cloud = cond.any(dim="CPR_height")

    # integer index of first True along CPR_height (argmax returns first max)
    idx_first = cond.argmax(dim="CPR_height")  # dtype=int, 0 if none True

    # prepare temperature DataArray in same (CPR_height, along_track) ordering
    var = ds[var_name]
    if set(("CPR_height", "along_track")).issubset(var.dims) and var.dims != (
        "CPR_height",
        "along_track",
    ):
        var = var.transpose("CPR_height", "along_track")

    # pick temperature at the CPR_height index for each time
    var_at_first_true = var.isel(CPR_height=idx_first.load())

    # mask out profiles that had no True so they become NaN
    var_at_first_true = var_at_first_true.where(has_cloud, other=np.nan)

    return var_at_first_true.assign_attrs(
        {
            "long_name": f"{var_name} at max height where {dbz_name} > {threshold_dbz} dBZ",
            "units": ds[var_name].attrs.get("units"),
            "dbz_threshold": threshold_dbz,
        }
    )


def get_difference_by_dbzs(ds, threshold_dbzs, var_name):
    var_at_max_heights = []
    for threshold_dbz in threshold_dbzs:
        var_at_max_heights.append(
            get_var_at_max_height_by_dbz(
                ds, threshold_dbz=threshold_dbz, var_name=var_name
            )
        )
    thickness = var_at_max_heights[0] - var_at_max_heights[1]
    return thickness.assign_attrs(
        {
            "long_name": f"{var_name} Difference at max heights where {threshold_dbzs[0]} dBZ and {threshold_dbzs[1]} dBZ",
            "units": ds[var_name].attrs.get("units"),
            "dbz_thresholds": threshold_dbzs,
        }
    )


def get_ml_xy(
    ds,
    feature_cols=["reflectivity_corrected", "temperature", "height"],
    return_mask=False,
):
    X = (
        ds.reset_coords()[feature_cols]
        .transpose("along_track", "CPR_height")
        .to_array()
        .stack(features=["variable", "CPR_height"])
        .load()
        .data
    )
    y = ds.reset_coords()["msi"].transpose("along_track", "band").load().data

    mask = (np.isfinite(y).all(axis=1) if y.ndim > 1 else np.isfinite(y)) & (
        y < 500
    ).all(axis=1)
    X = X[mask]
    y = y[mask]

    if return_mask:
        return xgb.DMatrix(X, label=y), xr.DataArray(
            data=mask,
            dims=["along_track"],
            coords={"along_track": ds.along_track},
            attrs={
                "long_name": "Mask for valid ML samples",
                "description": "True where samples are valid for ML input",
            },
        )
    else:
        return xgb.DMatrix(X, label=y)


def calculate_conditional_probabilities(
    y_true, y_pred, bin_edges=np.arange(180, 280, 2), return_sample_size=False
):
    bin_means, _, binnumber = binned_statistic(
        y_true, y_pred, statistic=np.nanmean, bins=bin_edges
    )
    bin_per90, _, _ = binned_statistic(
        y_true, y_pred, statistic=lambda x: np.nanpercentile(x, 90), bins=bin_edges
    )
    bin_per10, _, _ = binned_statistic(
        y_true, y_pred, statistic=lambda x: np.nanpercentile(x, 10), bins=bin_edges
    )

    h_joint_test_pred, _, _ = np.histogram2d(
        y_true, y_pred, bins=bin_edges, density=True
    )
    h_test, _ = np.histogram(y_true, bins=bin_edges, density=True)
    h_conditional = h_joint_test_pred / h_test.reshape(-1, 1)
    h_conditional_nan = np.where(h_conditional > 0, h_conditional, np.nan)

    if return_sample_size:
        return (
            bin_edges,
            bin_means,
            bin_per90,
            bin_per10,
            h_test,
            h_conditional_nan,
            np.bincount(binnumber),
        )
    else:
        return bin_edges, bin_means, bin_per90, bin_per10, h_test, h_conditional_nan
