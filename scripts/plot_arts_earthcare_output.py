# %%
import os
from plotting import (
    load_arts_output_data,
    plot_arts_output_distribution,
    plot_conditional_panel,
)
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
from ir_earthcare import habit_std_list, psd_list, get_cloud_top_height, get_cloud_top_T

# ignore invalid value warnings
import warnings

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered in divide *"
)
save = False


# %% Plot total distribution of ARTS output

# plot_arts_output_distribution(habit_std_idx=0, psd_idx=0, save=False)
for i in range(len(habit_std_list)):
    for j in range(len(psd_list)):
        plot_arts_output_distribution(i, j, save=save)
        plt.close()  # Close the figure to avoid displaying it in the notebook

# %% plot one orbit in time series
orbit_frame = "04015F"
habit_std, psd, orbits, ds_arts = load_arts_output_data(2, 2, orbit_frame=orbit_frame)
ds_arts = get_cloud_top_T(ds_arts, fwc_threshold=2e-5)

fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 8), constrained_layout=True)

# plot the dBZ, FWC, and LWC profiles
kwargs = dict(
    y="height_grid",
    x="nray",
    add_colorbar=True,
)
ds_arts["dBZ"].where(ds_arts["dBZ"] > -30).plot(ax=axes[0], vmin=-30, vmax=30, **kwargs)
# ds_arts["temperature"].plot(ax=axes[0], vmin=200, vmax=300, **kwargs)
ds_arts["bulkprop"].sel(scat_species="FWC").pipe(np.log10).plot(
    ax=axes[1], vmin=-7, vmax=-1, **kwargs
)
ds_arts["bulkprop"].sel(scat_species="LWC").pipe(np.log10).plot(
    ax=axes[2], vmin=-7, vmax=-1, **kwargs
)
ds_arts["surfaceElevation"].plot(
    ax=axes[0],
    x="nray",
    label="surface elevation",
    color="k",
    lw=1,
    ls="--",
    add_legend=True,
)
ds_arts["cloud_top_height"].plot(
    ax=axes[1],
    x="nray",
    label="cloud top height",
    color="C1",
    lw=1,
    ls="--",
    add_legend=True,
)
axes[0].legend(loc="center left")
axes[1].legend(loc="lower left")

# plot the brightness temperature
kwargs = dict(x="nray", marker=".", markersize=0.8, ls="-", lw=0.5, ax=axes[3])
ds_arts["arts"].mean("f_grid").plot(label="arts", **kwargs)
ds_arts["cloud_top_T"].plot(label="cloud top T", **kwargs)
ds_arts["pixel_values"].plot(label="MSI", **kwargs)
axes[3].legend(loc="center left")

# plot the difference between ARTS and MSI brightness temperature
kwargs = dict(
    x="nray", marker=".", markersize=0.8, ls="-", lw=0.5, ax=axes[4], ylim=[-20, 20]
)
# ds_arts[["pixel_values", "cloud_top_T"]].to_array().diff("variable").assign_attrs(units="K").plot(label="cloud top T", **kwargs)
ds_arts[["pixel_values", "arts"]].to_array().diff("variable").mean(
    "f_grid"
).assign_attrs(units="K").plot(label="arts", **kwargs)
axes[4].axhline(0, color="k", ls="--", lw=0.5)
axes[4].legend(loc="center left")

axes[0].set_title("Radar Reflectivity (dBZ)")
axes[1].set_title("FWC (log10)")
axes[2].set_title("LWC (log10)")
axes[3].set_title("Brightness Temperature (mean over f_grid)")
axes[4].set_title("Diff (X - MSI)")
axes[0].set_xlabel("")
axes[1].set_xlabel("")
axes[2].set_xlabel("")
axes[3].set_xlabel("")
axes[0].set_ylabel("Height [m]")
axes[1].set_ylabel("Height [m]")
axes[2].set_ylabel("Height [m]")

fig.suptitle(
    f"""
    {habit_std}
    {psd}
    {len(ds_arts.nray)} profiles
    """,
)
# put some text under the figure
plt.figtext(
    0.5,
    -0.15,
    f"""
    This figure shows the results of the ARTS simulation for EarthCARE data.
    The first three panels show the T, FWC (log10), and LWC (log10) profiles.
    The fourth panel shows the brightness temperature from ARTS and MSI.
    The last panel shows the difference between ARTS and MSI brightness temperature.
    The cloud top T is defined as the T at the highest height where the FWC is above a threshold ({ds_arts['cloud_top_T'].attrs['fwc_threshold']} kgm-3).
""",
    ha="center",
    fontsize=10,
    wrap=True,
)
# plt.show()
if save:
    plt.savefig(
        f"../data/figures/arts_output_series_{habit_std}_{psd}_{orbit_frame}.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(
        f'Figure is saved to "../data/figures/arts_output_series_{habit_std}_{psd}_{orbit_frame}.png"'
    )

#%%
orbit_frame = "04015F"

habits = []
psds = []
arts_T = []
arts_top_height = []
# Loop through all habit_std and psd combinations
j=0 # psd index
for i in range(3): # habit index
    habit_std, psd, orbits, ds_arts = load_arts_output_data(i, j, orbit_frame=orbit_frame)
    ds_arts = get_cloud_top_height(ds_arts, fwc_threshold=2e-5)
    habits.append(habit_std)
    psds.append(psd)
    arts_T.append(ds_arts['arts'].mean('f_grid'))
    arts_top_height.append(ds_arts['cloud_top_height'])

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 4), constrained_layout=True)

# plot the dB profiles
kwargs = dict(
    y="height_grid",
    x="nray",
    add_colorbar=True,
)
ds_arts["dBZ"].where(ds_arts["dBZ"] > -30).plot(ax=axes[0], vmin=-30, vmax=30, **kwargs)

ds_arts["surfaceElevation"].plot(
    ax=axes[0],
    x="nray",
    label="surface elevation",
    color="k",
    lw=1,
    ls="--",
    add_legend=True,
)
arts_top_height[0].plot(
    ax=axes[0],
    x="nray",
    label="cloud top height",
    color="C1",
    lw=1,
    ls="--",
    add_legend=True,
)
# axes[0].legend(loc="center left")

# plot the brightness temperature
kwargs = dict(ax=axes[1], x="nray", marker=".", markersize=0.8, ls="-", lw=0.5, ylim=[200, 280])
[a.plot(label=f"{h}", **kwargs) for h, a in zip(habits, arts_T)]
ds_arts["pixel_values"].plot(label="MSI", c='k', lw='0.7', **{k: v for k, v in kwargs.items() if k != 'lw'})

# plot the difference between ARTS and MSI brightness temperature
kwargs = dict(
    ax=axes[2], x="nray", marker=".", markersize=0.8, ls="-", lw=0.5, ylim=[-25, 25]
)
[(a-ds_arts['pixel_values']).assign_attrs(units="K").plot(label=f"{h}", **kwargs) for h, a in zip(habits, arts_T)]
axes[2].axhline(0, color="k", ls="--", lw=0.5)
axes[2].legend(loc="center left", bbox_to_anchor=(0.02, 0.5))

axes[0].set_title("Radar Reflectivity (dBZ)")
axes[1].set_title("Brightness Temperature (mean over f_grid)")
axes[2].set_title("Diff (ARTS - MSI)")
axes[0].set_xlabel("")
axes[1].set_xlabel("")
axes[2].set_xlabel("")
axes[0].set_ylabel("Height [m]")
axes[1].set_ylabel("Height [m]")
axes[2].set_ylabel("Height [m]")

fig.suptitle(
    f"""
    {psd}
    {len(ds_arts.nray)} profiles
    """,
)
if save:
    plt.savefig(
        f"../data/figures/arts_output_simple_series_{habit_std}_{psd}_{orbit_frame}.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(
        f'Figure is saved to "../data/figures/arts_output_simple_series_{habit_std}_{psd}_{orbit_frame}.png"'
    )
# %% groupby latitude bins
habit_std, psd, orbits, ds_arts = load_arts_output_data(2, 2)
ds_arts = get_cloud_top_T(ds_arts, fwc_threshold=1e-5)
plot_cloud_top_T = False  # Set to False to plot ARTS brightness temperature instead

# Group by latitude bins
groups = ds_arts.groupby_bins(
    abs(ds_arts["latitude"]),
    bins=[0, 30, 60, 90],
    labels=["low", "mid", "high"],
)

n_groups = len(groups)

fig, axes = plt.subplots(
    1,
    n_groups + 1,
    figsize=(6 * (n_groups + 1), 6),
    constrained_layout=True,
    sharex=True,
    sharey=True,
)
if n_groups + 1 == 1:
    axes = np.array([axes])

vmin, vmax = 0, 0.14
bin_edges_global = np.arange(180, 280, 2)


# Total (all data)
y_true_total = ds_arts["pixel_values"].values.flatten()
if plot_cloud_top_T:
    y_pred_total = ds_arts["cloud_top_T"].values.flatten()
else:
    y_pred_total = ds_arts["arts"].mean("f_grid").values.flatten()

c = plot_conditional_panel(
    ax=axes[0],
    y_true=y_true_total,
    y_pred=y_pred_total,
    bin_edges=bin_edges_global,
    title=f"Total\n{habit_std}, {psd}, {len(ds_arts.nray)} profiles",
    vmin=vmin,
    vmax=vmax,
    show_mbe=True,
)

# Latitude groups
for i, (group_name, ds_group) in enumerate(groups):
    y_true_group = ds_group["pixel_values"].values.flatten()
    if plot_cloud_top_T:
        y_pred_group = ds_group["cloud_top_T"].values.flatten()
    else:
        y_pred_group = ds_group["arts"].mean("f_grid").values.flatten()

    plot_conditional_panel(
        ax=axes[i + 1],
        y_true=y_true_group,
        y_pred=y_pred_group,
        bin_edges=bin_edges_global,
        title=f"{group_name.capitalize()} latitude\n{habit_std}, {psd}, {len(ds_group.nray)} profiles",
        vmin=vmin,
        vmax=vmax,
        show_mbe=True,
    )

fig.colorbar(
    c,
    ax=axes,
    orientation="vertical",
    label="P(Predicted | True)",
    fraction=0.02,
    pad=0.04,
)

if plot_cloud_top_T:
    plt.figtext(
        0.5,
        -0.15,
        f"""
        This figure shows the conditional PDF of the cloud top T (predicted) given the MSI brightness temperature (true).
        Cloud top T is defined as the T at the highest height where the FWC is above a threshold ({ds_arts['cloud_top_T'].attrs['fwc_treshhold']}).
        """,
        ha="center",
        fontsize=10,
        wrap=True,
    )

else:
    plt.figtext(
        0.5,
        -0.15,
        f"""
        This figure shows the conditional PDF of the ARTS brightness temperature (predicted) given the MSI brightness temperature (true).
        The first panel shows the total distribution, while the other panels show the distribution for different latitude bins.
        The assumed habit and PSD, and the total number of orbit frames are shown in the title.
        The ARTS brightness temperature is averaged over the f_grid (length of {len(ds_arts['f_grid'])}).
        """,
        ha="center",
        fontsize=10,
        wrap=True,
    )

    if save:
        plt.savefig(
            f"../data/figures/arts_output_distribution_latitude_{habit_std}_{psd}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(
            f'Figure is saved to "../data/figures/arts_output_distribution_latitude_{habit_std}_{psd}.png"'
        )


# %% read an input file
# path = f"../data/earthcare/arts_input_data/arts_input_{'03645E'}.nc"
# ds = xr.open_dataset(path)

# fig, ax = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True, sharex=True)
# ds.dBZ.plot(ax=ax[0], x="nray", y="height_grid", vmin=-30, vmax=30, add_colorbar=True)
# ds.pixel_values.plot(ax=ax[1], x="nray", marker=".", markersize=1, label="MSI")
