# %%
# %load_ext autoreload
# %autoreload 2
from ectools import ecio, ecplot, colormaps
from matplotlib.colors import LogNorm, NoNorm
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
import psd

# %% load and merge datasets
orbit_frame = "06356H"
ds_acm = ecio.load_ACMCAP(
    srcpath=dp.ACMCAP,
    product_baseline="BA",
    orbit=orbit_frame[:-1],
    frame=orbit_frame[-1],
    nested_directory_structure=True,
).reset_coords()
ds_acm.close()

ds_xmet = ecio.load_XMET(
    dp.XMET,
    orbit=orbit_frame[:-1],
    frame=orbit_frame[-1],
)
ds_xmet.close()
ds_acm = ecio.get_XMET(
    ds_xmet,
    ds_acm,
    XMET_1D_variables=[],
    XMET_2D_variables=["temperature", "pressure", "specific_humidity"],
)

ds_acm = ds_acm.rename(
    {
        "height": "height_acm",
        "MSI_longwave_channel": "band",
        "temperature": "temperature_acm",
    }
)
ds = ds_acm
# %% load ARTS results
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

# %% Plot reflectivity and residuals
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


plt_kws = {"alpha": 0.7, "linewidth": 1, "markersize": 5}
plot1D_dict = {
    f"ARTS_{p}": {
        "xdata": ds["time"],
        "ydata": ds["residual_arts"].sel(habit=Habit.Column, psd=p).isel(band=1),
        **plt_kws,
    }
    for p in [PSD.D14, PSD.F07T]
}


plot1D_dict.update(
    {
        "ACM_CAP": {
            "xdata": ds["time"],
            "ydata": ds["residual_acmcap"].isel(band=1),
            **plt_kws,
        },
        "XGBoost": {
            "xdata": ds["time"],
            "ydata": ds["residual_xgb"].isel(band=1),
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

# make a star marker on a selected time/height point
if False:
    along_track = ds.isel(along_track=2900).along_track.data.item()
    height = 10e3  # m
    t_marker = ds["time"].sel(along_track=along_track).data
    ax[0, 0].plot(
        t_marker,
        height,
        marker="*",
        markersize=30,
        markerfacecolor="white",
        markeredgecolor="k",
        zorder=10,
    )

# %% Get FWC from onion inversion table
from onion_table import make_onion_invtable

fwcs = []
for psd_type in [PSD.D14, PSD.F07T, PSD.MDG]:
    for habit in ["LargeColumnAggregate", Habit.Plate]:
        ds_onion_table = make_onion_invtable(
            habit=habit,
            psd=psd_type,
            coef_mgd=(
                {
                    "n0": 1e10,  # Number concentration
                    "ga": 1.5,  # Gamma parameter
                    "mu": 0,  # Default value for mu
                }
                if psd_type == PSD.MDG
                else None
            ),
            return_xarray=True,
        )
        fwc = (
            ds_onion_table.sel(radiative_properties="FWC")
            .interp(
                Temperature=ds["temperature_acm"],
                dBZ=ds["CPR_reflectivity_factor"],
            )
            .pipe(lambda x: 10**x)  # Convert from log10(FWC) to FWC in kg/m^3
            .fillna(0)
            .expand_dims(["psd", "habit"])
            .assign_coords({"psd": [psd_type], "habit": [habit]})
            .rename("ice_water_content_onion")
            .assign_attrs(ds["ice_water_content"].attrs)
            .assign_attrs({"long_name": f"IWC from Onion Inversion Table"})
        )
        fwcs.append(fwc)

fwc = xr.combine_by_coords(fwcs, combine_attrs="override")
ds = ds.merge(fwc)

# %% make plots of relative difference between onion and acmcap iwc
ds["iwc_difference"] = (
    (ds["ice_water_content_onion"] - ds["ice_water_content"])
    / ds["ice_water_content"].where(ds["ice_water_content"] > 0)
    * 100
)
psd_to_plot = [PSD.D14, PSD.F07T]
habit_to_plot = ["LargeColumnAggregate"]
nrows = len(psd_to_plot) + 1
ncols = len(habit_to_plot)
fig, ax = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(25 * ncols, 7 * nrows),
    squeeze=False,
    sharex=False,
    sharey=True,
)
for i, psd_type in enumerate(psd_to_plot):
    for j, habit in enumerate(habit_to_plot):
        ecplot.plot_EC_2D(
            ax=ax[i, j],
            ds=ds.sel(psd=psd_type, habit=habit),
            varname="iwc_difference",
            heightvar="height_acm",
            label="IWC Rel. Diff.",
            units="%",
            plot_range=[-100, 100],
            plot_scale="linear",
            title=f"IWC Rel. Difference (Onion - ACM CAP) {psd_type} - {habit}",
            cmap=plt.get_cmap("RdBu_r"),
            product_label=f"{psd_type} - {habit}",
        )

# add original acmcap iwc plot
for i in range(ncols):
    ecplot.plot_EC_2D(
        ax=ax[-1, i],
        ds=ds,
        varname="ice_water_content",
        heightvar="height_acm",
        label="IWC",
        title="IWC from ACM CAP Retrieval",
        cmap=colormaps.calipso,
    )
ecplot.add_marble(ax[0, 0], ds)
fig.tight_layout()

# %% Plot PSD at a specific time and height
from easy_arts.size_dists import dgeo2deq
from easy_arts import scat_data, size_dists

habit = "LargeColumnAggregate"
along_track_idx = 3800
JSG_height_idx = 150
height = ds['height_acm'].isel(along_track=along_track_idx, JSG_height=JSG_height_idx).data.item()
# height = 11e3  # m
# height_at_along_track = ds["height_acm"].isel(along_track=along_track_idx)
# sorter = np.argsort(height_at_along_track)
# idx_sorter = np.searchsorted(height_at_along_track, height, sorter=sorter)
# JSG_height_idx = sorter[idx_sorter]

a, b, rho = 0.21, 2.26, 1000  # values in plate aggregate from database
d_max = np.logspace(-6, -2, 50)  # m
d_meq = dgeo2deq(d_max, a=a, b=b, rho=rho)  # m

t = (
    ds["temperature_acm"]
    .isel(along_track=along_track_idx, JSG_height=JSG_height_idx)
    .load()
    .pipe(np.atleast_1d)
)

iwc_onion = (
    ds["ice_water_content_onion"]
    .isel(along_track=along_track_idx, JSG_height=JSG_height_idx)
    .sel(habit=habit)
    .load()
)

da_psd_values = []
psd_list = [PSD.D14, PSD.F07T, PSD.MDG, "ACM_CAP"]
for i, psd_type in enumerate(psd_list):
    if psd_type == "ACM_CAP":
        n0star = (
            ds["ice_normalized_number_concentration"]
            .isel(along_track=along_track_idx, JSG_height=JSG_height_idx)
            .pipe(np.atleast_1d)
        )

        d0star = (
            ds["ice_median_volume_diameter"]
            .isel(along_track=along_track_idx, JSG_height=JSG_height_idx)
            .pipe(np.atleast_1d)
        )
        fwc = ds["ice_water_content"].isel(along_track=along_track_idx, JSG_height=JSG_height_idx).pipe(
            np.atleast_1d
        )
        r = (
            ds["ice_riming_factor"]
            .isel(along_track=along_track_idx, JSG_height=JSG_height_idx)
            .pipe(np.atleast_1d)
        )
        da_psd = psd.psd_ACMCAP(dmax=d_max, n0star=n0star, d0star=d0star).assign_coords(fwc=("input_setting", fwc))
        a, b = size_dists.acm_cap_am_bm(r=r)
        da_psd = psd.convert_psd_dgeo2deq(
            da_psd=da_psd,
            a=a,  # note that a and b are different when sizes are smaller than 100um! Not yet implemented
            b=b,
            varname_dgeo="d_max",
            varname_deq="d_meq",
        )
        da_psd = psd.align_psd_size_grids(
            da_psd=da_psd, d_name="d_meq", common_size_grid=d_meq
        )

    else:
        da_psd = psd.get_psd_dataarray(
            psd_size_grid=d_max if psd_type in [PSD.F07M, PSD.F07T, PSD.MDG] else d_meq,
            fwc=iwc_onion.sel(psd=psd_type).pipe(np.atleast_1d),  # kg/m^3
            t=t,  # K
            psd=psd_type,
            coef_mgd={"n0": 1e10, "ga": 1.5, "mu": 0},
            scat_species_a=a,
            scat_species_b=b,
            convert_to_deq=True,
            varname_deq="d_meq",
            varname_dgeo="d_max",
        )

    da_psd = psd.align_psd_size_grids(
        da_psd=da_psd, d_name="d_meq", common_size_grid=d_meq
    )
    da_psd_values.append(da_psd)


# %% backscatter calculation
from easy_arts import scat_data, size_dists

from physics.constants import DENSITY_H2O_ICE, DENSITY_H2O_LIQUID

# S, M = scat_data.from_xml(datafolder=dp.single_scattering_database_arts, habit=habit)
# ds_ss = scat_data.to_xarray(S, M, interpolate_na=False)
ds_ss = scat_data.import2xarray(
    datafolder=dp.single_scattering_database_arts, habit=habit
)

# xs_ext, xs_abs, phase_matrix = (
#     ds_ss["ext_mat_data"],
#     ds_ss["abs_vec_data"],
#     ds_ss["pha_mat_data"],
# )
phase_backscatter_94GHz = xr.DataArray(
    scat_data.backscattering(ds_ss, freq=94e9, temp=t),
    dims={"size": ds_ss.size},
    coords={"d_veq": ("size", ds_ss.d_veq.data)},
)
phase_backscatter_94GHz = (
    # (
    #     phase_matrix.sel(
    #         f_grid=94e9,
    #         T_grid=t,
    #         za_sca_grid=180,
    #         method="nearest",
    #     )
    #     .isel(pha_mat_element=0)
    #     .squeeze()
    # )
    phase_backscatter_94GHz
    # convert d_veq to d_meq
    .assign_coords(
        {
            "d_meq": size_dists.deq2deq(
                ds_ss["d_veq"],
                rho1=DENSITY_H2O_ICE,
                rho2=DENSITY_H2O_LIQUID,
            )
        }
    )
    .pipe(np.log)
    .swap_dims(size="d_meq")
    .interp(d_meq=d_meq, kwargs={"fill_value": "extrapolate"})
    .pipe(np.exp)
    .swap_dims(d_meq="size")
)  # m^2

backscatter_values = []
for da_psd in da_psd_values:
    backscatter_values.append(da_psd * phase_backscatter_94GHz)

# %% plotting
psd_list = [PSD.D14, PSD.F07T, PSD.MDG, "ACM_CAP"]
linestyles = ["-", "--", "-.", ":"]
# colors = plt.get_cmap("tab10").colors  # colorblind-friendly palette
# colors = ["#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
colors = ["#e69f00", "#56b4e9", "#009e73", "#cc79a7"]
fig, ax = plt.subplots(3, 1, figsize=(12, 6 * 3), sharex=False)
for i, (da_psd, backscatter) in enumerate(zip(da_psd_values, backscatter_values)):
    plot_kws = {
        "linewidth": 8,
        "linestyle": linestyles[i % len(linestyles)],
        "color": colors[i % len(colors)],
        "alpha": 0.9,
        "x": "d_meq",
        "xscale": "linear",
    }
    psd_type = da_psd.attrs["psd_type"]
    da_psd.plot(ax=ax[0], label=f"{psd_type}", yscale="log", **plot_kws)
    backscatter.plot(ax=ax[1], label=f"{psd_type}", **plot_kws)

    ax[2].scatter(
        da_psd.fwc.data.item(),
        backscatter.integrate("d_meq").item(),
        color=colors[i % len(colors)],
        s=2000,
        alpha=0.9,
    )
    ax[2].text(
        da_psd.fwc.data.item(),
        backscatter.integrate("d_meq").item(),
        psd_type,
        fontsize=20,
        ha="center",
        va="top",
    )

ax[0].set_title(
    f"PSD at along_track index {along_track_idx}\nand height {height*1e-3:.1f} km\nT={t.item():.1f} K"
)
ax[0].set_xlabel(r"$D_{meq}$ (m)")
ax[0].set_ylabel(r"$N(D_{meq})$ (m$^{-4}$)")
ax[0].set_ylim(1e0, 1e12)
ax[0].set_xlim(1e-5, 1e-3)
ax[0].legend(title=f"{habit}")

ax[1].set_title("Radar backscatter")
ax[1].set_xlabel(r"$D_{meq}$ (m)")
ax[1].set_ylabel(r"$\phi_{180} \cdot N(D_{meq})$ (m$^{-4} sr^{-1}$)")
ax[1].legend(title=f"{habit}")
ax[1].sharex(ax[0])
ax[1].set_ylim(0, 1.4e-3)

ax[2].set_xlabel("IWC (kg/m3)")
ax[2].set_ylabel("$\\int \phi_{180} \cdot N(D_{meq}) dD_{meq}$ (m$^{-1}$)")
ax[2].set_xscale("linear")
ax[2].set_yscale("linear")
# ax[2].set_ylim(0, 3e-8)
# ax[2].set_xlim(0, 6e-5)
ax[2].set_title(habit)


# hide second plot
# ax[1].set_visible(False)
plt.tight_layout()

# %% plot phase matrix for different habits
fig, ax = plt.subplots(1, 1, figsize=(12, 6), squeeze=False)
for h in [Habit.Column, Habit.Bullet, Habit.Plate]:
    S, M = scat_data.from_xml(datafolder=dp.single_scattering_database_arts, habit=h)
    ds_ss = scat_data.to_xarray(S, M, interpolate_na=False)
    phase_matrix = ds_ss["pha_mat_data"]
    p = (
        phase_matrix.sel(
            f_grid=95e9,
            za_sca_grid=180,
            T_grid=t,
            method="nearest",
        )
        .isel(element=0)
        .assign_coords({"d_meq": phase_matrix["d_veq"] * 0.97})
    )
    p.plot(
        ax=ax[0],
        x="d_meq",
        xscale="linear",
        yscale="log",
        label=h,
        add_legend=True,
    )

ax[0].set_title(
    f"Phase matrix element P11 at {p.f_grid.item()*1e-9:.1f} GHz, {p.T_grid.item():.1f} K, {p.za_sca_grid.item()}° scattering angle"
)
ax[0].set_xlabel(r"$D_{meq}$ (m)")
ax[0].set_ylabel(r"$\phi_{180}$ (sr$^{-1}$)")
ax[0].legend(title="Habit")
ax[0].sharex(ax[0])
ax[0].set_xlim(1e-5, 1e-3)
ax[0].set_ylim(1e-14, 1e-7)

# %%

# %% 2D histogram of N0star vs Temperature
skip_n_profiles = 20
hist, xedges, yedges = np.histogram2d(
    x=ds["temperature"]
    .isel(along_track=slice(None, None, skip_n_profiles))
    .pipe(lambda x: x - 273.15)
    .stack(data=["along_track", "JSG_height"])
    .load()
    .data,
    y=ds["ice_N0prime"]
    .isel(along_track=slice(None, None, skip_n_profiles))
    .pipe(np.log10)
    .stack(data=["along_track", "JSG_height"])
    .load()
    .data,
    bins=[np.arange(-80, 0, 2), np.arange(2, 13, 0.2)],
)
# %%
from psd import get_n0star_from_temperature, get_n0prime_from_temperature
from scipy.optimize import curve_fit

plt.figure(figsize=(10, 6))
plt.pcolormesh(yedges, xedges, hist, cmap="Blues")
plt.colorbar(label="Counts")
plt.ylabel("Temperature (°C)")
# plt.xlabel("$\log_{10}(N_0^*)$ ($m^{-4}$)")
plt.xlabel("$\log_{10}(N_0^')$ ($m^{-3.4}$)")
plt.title("2D Histogram of N0' vs T")
plt.gca().invert_yaxis()

plt.plot(
    np.log10(get_n0prime_from_temperature(xedges + 273.15)),
    xedges,
    color="red",
    # label="A priori ($N_0^'(T_C)/a_v^{0.6}(T_C)$)",
    label="A priori $N_0^'(T_C)$",
)


def linear_model(t, a, b):
    return a - b * t


# Prepare data for fitting
T_fit = xedges[:-1] + np.diff(xedges) / 2  # bin centers
N0_log_fit = ((yedges[:-1] + np.diff(yedges) / 2) * hist).sum(axis=1) / hist.sum(
    axis=1
)  # weighted average log10(N0*)
mask = (N0_log_fit > 0) & (T_fit > -60) & (T_fit < -20)
popt, _ = curve_fit(linear_model, T_fit[mask], N0_log_fit[mask])
a_fit, b_fit = popt

# add fitted curve to your plot
plt.plot(
    linear_model(T_fit, *popt),
    T_fit,
    ls="--",
    color="red",
    label=f"Fitted: {a_fit:.2f} - {b_fit:.3f} $ T_C$",
)
plt.plot(
    N0_log_fit[mask],
    T_fit[mask],
    color="lightgrey",
    marker="*",
    ls="",
    label="Data Points for Fit \n(bin average)",
)
plt.legend(fontsize=15, loc="upper left")
plt.show()

# %%
