# %%
from ectools import ecio, ecplot
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import data_paths as dp
from analysis import (
    load_arts_results,
    load_and_merge_acmcap_data,
    load_and_merge_ec_data,
    get_bias,
    get_ml_xy,
)
from earthcare_ir import Habit, PSD
import pandas as pd

# %%

ds_acm = (
    ecio.load_ACMCAP(
        srcpath=dp.ACMCAP,
        product_baseline="BA",
        orbit="06356",
        frame="H",
        nested_directory_structure=True,
    )
    .swap_dims(along_track="time")
    .reset_coords()
)
ds_acm.close()
ds_acm = ds_acm.rename({"height": "height_acm", "MSI_longwave_channel": "band"})

orbit_frame = ds_acm.encoding["source"].split("/")[-1].split("_")[-1].split(".")[0]
ds_arts = load_arts_results(orbit_frame=orbit_frame)
ds_arts.close()

ds_arts_ec = load_and_merge_ec_data(ds_arts)
ds_arts_ec.close()

# merge acm and arts datasets
ds_arts_re = ds_arts_ec.swap_dims(along_track="time").reindex_like(
    ds_acm, method="nearest", tolerance=pd.Timedelta("1s")
)
ds = ds_arts_re.merge(
    ds_acm,
    overwrite_vars=[
        "latitude",
        "longitude",
        "along_track",
    ],
).swap_dims(time="along_track")
ds.encoding = ds_acm.encoding

# %% XGBoost
import xgboost as xgb

model = xgb.Booster()
model_tag = "second_try"
model.load_model(os.path.join(dp.XGBoost, f"xgboost_model_{model_tag}.json"))
dtest, mask = get_ml_xy(ds, return_mask=True)
if dtest.num_row() != ds.sizes["along_track"]:
    print("Mismatch in number of samples between dtest and ds")
    ds = ds.where(mask, drop=True)
y_pred = model.predict(dtest)
ds = ds.assign({"xgb": (("along_track", "band"), y_pred)})
ds.encoding = ds_acm.encoding

# %%
ds["residual_acmcap"] = get_bias(
    ds=ds,
    var_name1="MSI_longwave_brightness_temperature",
    var_name2="MSI_longwave_brightness_temperature_forward",
)
ds["residual_arts"] = get_bias(ds=ds, var_name1="msi", var_name2="arts")
ds["residual_xgb"] = get_bias(ds=ds, var_name1="msi", var_name2="xgb")

# %%
nrows = 2
fig, ax = plt.subplots(
    nrows=nrows, ncols=1, figsize=(25, 7 * nrows), squeeze=False, sharex=False
)
ecplot.plot_EC_2D(
    ax=ax[0, 0],
    ds=ds,
    varname="CPR_reflectivity_factor",
    heightvar="height_acm",
    label="Z",
    units="dBZ",
    # plot_scale="linear",
    plot_range=[-30, 30],
)


plt_kws = {"alpha": 0.5, "linewidth": 1, "markersize": 5}
plot1D_dict = {
    f"ARTS_{p}": {
        "xdata": ds["time"],
        "ydata": ds["residual_arts"].sel(habit=Habit.Column, psd=p).isel(band=1),
        **plt_kws,
    }
    for p in [PSD.D14, PSD.F07T, PSD.MDG]
}


plot1D_dict.update(
    {
        "XGBoost": {
            "xdata": ds["time"],
            "ydata": ds["residual_xgb"].isel(band=1),
            **plt_kws,
        },
        "ACM_CAP": {
            "xdata": ds["time"],
            "ydata": ds["residual_acmcap"].isel(band=1),
            **plt_kws,
        },
    }
)

ecplot.plot_EC_1D(
    ax=ax[1, 0],
    ds=ds,
    plot1D_dict=plot1D_dict,
    ylabel="MSI - X (K)",
    title="Residuals of TIR2 Brightness Temperature",
    include_ruler=False,
)

ax[1, 0].axhline(0, color="k", linestyle="--", linewidth=1)
plt.tight_layout()

# %% N0prime

nrows = 1
fig, ax = plt.subplots(
    nrows=nrows, ncols=1, figsize=(25, 7 * nrows), squeeze=False, sharex=False
)

ecplot.plot_EC_2D(
    ax=ax[0, 0],
    ds=ds_acm,
    varname="ice_N0prime",
    label="N0'",
)
