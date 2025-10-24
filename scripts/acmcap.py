# %%
from ectools import ecio, ecplot
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import data_paths as dp
from plotting import load_arts_results
from earthcare_ir import Habit, PSD
import pandas as pd

# %%
ds_acm = (
    ecio.load_ACMCAP(
        srcpath=dp.ACMCAP,
        product_baseline="BA",
        frame_code="06356H",
        nested_directory_structure=True,
    )
    .swap_dims(along_track="time")
    .reset_coords()
)
ds_acm.close()

orbit_frame = ds_acm.encoding["source"].split("/")[-1].split("_")[-1].split(".")[0]
ds_arts = (
    load_arts_results(orbit_frame=orbit_frame)
    .swap_dims(along_track="time")
    .reset_coords()
)
ds_arts.close()

# merge acm and arts datasets
ds_arts_re = ds_arts.reindex_like(
    ds_acm, method="nearest", tolerance=pd.Timedelta("1s")
)#.assign({var: ds_acm[var] for var in ['latitude', 'longitude', 'along_track']})
ds = ds_arts_re.merge(ds_acm, overwrite_vars=["latitude", "longitude", "along_track"]).swap_dims(time='along_track').reset_coords()
ds.encoding = ds_acm.encoding

# %%
nrows = 2
fig, ax = plt.subplots(
    nrows=nrows, ncols=1, figsize=(25, 7 * nrows), squeeze=False, sharex=False
)
ecplot.plot_EC_2D(
    ax=ax[0, 0],
    ds=ds,
    varname="CPR_reflectivity_factor",
    label="Z",
    units="dBZ",
    # plot_scale="linear",
    plot_range=[-30, 30],
)


plot1D_dict={
    f"ARTS_{p}": {
        "xdata": ds["time"],
        "ydata": (
            ds["arts"].sel(habit=Habit.Column, psd=p)
            - ds["MSI_longwave_brightness_temperature"]
        ).isel(MSI_longwave_channel=1),
        "alpha": 0.5,
    }
    for p in [PSD.D14, PSD.F07T, PSD.MDG]
    }

plot1D_dict.update({
        "ACM_CAP": {
            "xdata": ds["time"],
            "ydata": (
                ds["MSI_longwave_brightness_temperature_forward"]
                - ds["MSI_longwave_brightness_temperature"]
            ).isel(MSI_longwave_channel=1),
            "alpha": 0.5,
        },
    }
)

ecplot.plot_EC_1D(
    ax=ax[1, 0],
    ds=ds,
    plot1D_dict=plot1D_dict,
    ylabel="X - MSI (K)",
    title="Residuals of TIR2 Brightness Temperature",
    include_ruler=False,
)

ax[1,0].axhline(0, color='k', linestyle='--', linewidth=1)
plt.tight_layout()

# %% N0prime

nrows = 1
fig, ax = plt.subplots(
    nrows=nrows, ncols=1, figsize=(25, 7 * nrows), squeeze=False, sharex=False
)

ecplot.plot_EC_2D(
    ax=ax[0, 0],
    ds=ds,
    varname="ice_N0prime",
    label="N0'",
)