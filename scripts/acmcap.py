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

# %% load and merge datasets
ds_acm = ecio.load_ACMCAP(
    srcpath=dp.ACMCAP,
    product_baseline="BA",
    orbit="06356",
    frame="H",
    nested_directory_structure=True,
).reset_coords()
ds_acm.close()
ds_acm = ds_acm.rename({"height": "height_acm", "MSI_longwave_channel": "band"})

orbit_frame = ds_acm.encoding["source"].split("/")[-1].split("_")[-1].split(".")[0]
ds_arts = load_arts_results(orbit_frame=orbit_frame)
ds_arts.close()

ds_arts_ec = load_and_merge_ec_data(ds_arts)
ds_arts_ec.close()

# merge acm and arts datasets
ds_arts_re = ds_arts_ec.swap_dims(along_track="time").reindex_like(
    ds_acm.swap_dims(along_track="time"), method="nearest", tolerance=pd.Timedelta("1s")
)
ds = ds_arts_re.merge(
    ds_acm.swap_dims(along_track="time"),
    overwrite_vars=[
        "latitude",
        "longitude",
        # "along_track",
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
    var_name1="MSI_longwave_brightness_temperature_forward",
    var_name2="MSI_longwave_brightness_temperature",
)
ds["residual_arts"] = get_bias(ds=ds, var_name1="arts", var_name2="msi")
ds["residual_xgb"] = get_bias(ds=ds, var_name1="xgb", var_name2="msi")

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
    ylabel="X - MSI (K)",
    title="Residuals of TIR2 Brightness Temperature",
    include_ruler=False,
)

ax[1, 0].axhline(0, color="k", linestyle="--", linewidth=1)

ecplot.add_marble(ax[0, 0], ds)
plt.tight_layout()

# %% Onion inversion table
from onion_table import get_ds_onion_invtable

fwcs = []
for psd in [PSD.D14, PSD.F07T, PSD.MDG]:
    for habit in [Habit.Column]:
        ds_onion_table = get_ds_onion_invtable(
            habit=habit,
            psd=psd,
            coef_mgd=(
                {
                    "n0": 1e10,  # Number concentration
                    "ga": 1.5,  # Gamma parameter
                    "mu": 0,  # Default value for mu
                }
                if psd == PSD.MDG
                else None
            ),
        )
        fwc = (
            ds_onion_table.sel(radiative_properties="FWC")
            .interp(
                Temperature=ds["temperature"],
                dBZ=ds.sel(psd=psd, habit=habit)["reflectivity_corrected"],
            )
            .pipe(lambda x: 10**x)  # Convert from log10(FWC) to FWC in kg/m^3
            .fillna(0)
            .expand_dims(["psd", "habit"])
        )
        fwcs.append(fwc)

fwc = xr.concat(fwcs, dim="psd").assign_attrs(ds["ice_water_content"].attrs)
ds = ds.assign({"ice_water_content_onion": fwc})

# %% compare IWC
fig, ax = plt.subplots(
    nrows=4, ncols=1, figsize=(25, 7 * 4), squeeze=False, sharex=False
)

ecplot.plot_EC_2D(
    ax=ax[0, 0],
    ds=ds,
    varname="ice_water_content",
    heightvar="height_acm",
    label="ACMCAP IWC",
    units="kg/m$^3$",
)

for i, psd in enumerate([PSD.D14, PSD.F07T, PSD.MDG]):
    ecplot.plot_EC_2D(
        ax=ax[i + 1, 0],
        ds=ds.sel(psd=psd, habit=Habit.Column),
        varname="ice_water_content_onion",
        heightvar="height",
        label=psd + " IWC",
        title=f"IWC from Onion Inversion Table - {psd}",
    )


ax[-1, 0].axhline(0, color="k", linestyle="--", linewidth=1)
ecplot.add_marble(ax[0, 0], ds)
plt.tight_layout()

# for i in range(4):
#     ax[i, 0].axvline(ds["time"].sel(along_track=along_track).data, color="k", linestyle="--")
#     ax[i, 0].axhline(height, color="k", linestyle="--")

# %% N0prime and d0star calculation
from get_psd import get_d0star_from_n0star_iwc


ds["d0star"] = ds.pipe(
    lambda x: get_d0star_from_n0star_iwc(
        x["ice_water_content"],
        x["ice_normalized_number_concentration"],
    )
).assign_attrs(
    {
        "long_name": "Normalized median diameter",
        "units": "m",
        "plot_range": np.array([1e-5, 1e-2]),
        "plot_scale": "logarithmic",
    }
)


# %% Plot IWC, N0star, d0star
nrows = 3
fig, ax = plt.subplots(
    nrows=nrows, ncols=1, figsize=(25, 7 * nrows), squeeze=False, sharex=False
)

ecplot.plot_EC_2D(
    ax=ax[0, 0],
    ds=ds,
    varname="ice_water_content",
    heightvar="height_acm",
    label="IWC",
)

ecplot.plot_EC_2D(
    ax=ax[1, 0],
    ds=ds,
    varname="ice_normalized_number_concentration",
    heightvar="height_acm",
    label="N0*",
)

ecplot.plot_EC_2D(
    ax=ax[2, 0],
    ds=ds,
    varname="d0star",
    heightvar="height_acm",
    label="d0*",
)
ecplot.add_marble(ax[0, 0], ds)

ax[0, 0].axvline(ds["time"].isel(along_track=3000).data, color="k", linestyle="--")
ax[0, 0].axhline(10000, color="k", linestyle="--")

# %% Plot PSD at a specific time and height

from get_psd import get_psd, psd_F05_norm
from physics.unitconv import dgeo2deq

along_track = ds.isel(along_track=2900).along_track.data.item()
height = 8e3  # m

fig, ax = plt.subplots(1,2, figsize=(25, 10))
a, b, rho = 0.21, 2.26, 1000  # values in plate aggregate from database
d_max = np.logspace(-6, -2, 100)  # m
d_meq, ddmeq_ddmax = dgeo2deq(d_max, a=a, b=b, rho=rho)  # m

t = (
    (
        ds["temperature"]
        .sel(along_track=along_track)
        .dropna("CPR_height")
        .swap_dims(CPR_height="height")
        .sel(height=height, method="nearest")
    )
    .load()
    .data
)  # K

for psd in [PSD.D14, PSD.F07T, PSD.MDG, "ACM_CAP"]:
    if psd == PSD.F07M or psd == PSD.F07T or psd == PSD.MDG:
        d = d_max
    else:
        d = d_meq

    if psd == "ACM_CAP":
        d0star = (
            ds.set_coords("height_acm")["d0star"]
            .sel(along_track=along_track)
            .swap_dims(JSG_height="height_acm")
            .sel(height_acm=height, method="nearest")
        ).data  # m
        n0star = (
            ds.set_coords("height_acm")["ice_normalized_number_concentration"]
            .sel(along_track=along_track)
            .swap_dims(JSG_height="height_acm")
            .sel(height_acm=height, method="nearest")
        ).data  # m^-4
        iwc = (
            ds.set_coords("height_acm")["ice_water_content"]
            .sel(along_track=along_track)
            .swap_dims(JSG_height="height_acm")
            .sel(height_acm=height, method="nearest")
        ).data  # kg/m^3
        psd_data = n0star * psd_F05_norm(d_meq / d0star)  # m^-4
    else:
        iwc = (
            ds.set_coords("height")["ice_water_content_onion"]
            .sel(along_track=along_track, psd=psd, habit=Habit.Column)
            .where(~ds.height.sel(along_track=along_track).isnull().load(), drop=True)
            .swap_dims(CPR_height="height")
            .sel(height=height, method="nearest")
        ).load()  # .data  # kg/m^3
        _, psd_data = get_psd(
            fwc=np.atleast_1d(iwc),  # kg/m^3
            t=np.atleast_1d(t),  # K
            psd=psd,
            psd_size_grid=d,
            mgd_coef={"n0": 1e10, "ga": 1.5, "mu": 0},
            rho=rho,
        )
        psd_data = psd_data.value.squeeze()  # m^-4

    # make sure all psd are in terms of d_meq
    if psd == PSD.F07T or psd == PSD.F07M or psd == PSD.MDG:
        psd_data *= ddmeq_ddmax
    ax[0].loglog(d_meq, psd_data, marker=".", label=f'{psd}, IWC={iwc:.2e} kg/m^3')
    ax[1].semilogx(d_meq, psd_data*d_meq**6, marker=".", label=psd)

ax[0].set_title(
    f"PSD at along_track index {along_track}\nand height {height*1e-3} km\nT={t:.1f} K"
)
ax[0].set_xlabel("D")
ax[0].set_ylabel("N(D) (m^-4)")
ax[0].set_ylim(1e2, 1e12)
ax[0].legend()

ax[1].set_title('6th moment')
ax[1].set_xlabel("D")
ax[1].set_ylabel("N(D)*D^6")
ax[1].legend()

plt.tight_layout()