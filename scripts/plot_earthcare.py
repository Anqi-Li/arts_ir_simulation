# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import glob

# ignore invalid value warnings
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide *")
save = False


# %% read output files
def load_arts_earthcare_data(habit_std_idx, psd_idx, orbit_frame=None, n_files=None, random_seed=42):
    habit_std = ["LargePlateAggregate", "8-ColumnAggregate"][habit_std_idx]
    psd = ["DelanoeEtAl14", "FieldEtAl07TR", "FieldEtAl07ML"][psd_idx]
    if orbit_frame is None:
        orbit_frame = "*"  # Use wildcard to match all orbit frames

    pattern = (
        f"../data/earthcare/arts_output_data/cold_3rd_{habit_std}_{psd}_{orbit_frame}.nc"
        # f'/scratch/li/arts-earthcare/arts_output_data_old/cold_allsky_{habit_std}_{psd}_{orbit_frame}.nc'
    )
    matching_files = sorted(glob.glob(pattern))
    if n_files is not None and len(matching_files) > n_files:
        rng = np.random.default_rng(random_seed)
        matching_files = rng.choice(matching_files, size=n_files, replace=False)
        matching_files = matching_files.tolist()
    orbits = [o[-9:-3] for o in matching_files]
    print(f"Number of matching files: {len(matching_files)}")

    datasets = [xr.open_dataset(f, chunks='auto').assign_coords(orbit_frame=f[-9:-3]) for f in matching_files]
    ds_arts = xr.concat(datasets, dim="nray").sortby("profileTime")

    return habit_std, psd, orbits, ds_arts


def get_cloud_top_height(ds, fwc_threshold=1e-5):
    """Calculate the cloud top height based on the frozen water content."""
    return ds["height_grid"].where(ds["frozen_water_content"] > fwc_threshold).max("height_grid")


def get_cloud_top_T(ds, fwc_threshold=1e-5):
    ds["cloud_top_height"] = get_cloud_top_height(ds, fwc_threshold=fwc_threshold)
    return ds["temperature"].where(ds["height_grid"] == ds["cloud_top_height"]).min("height_grid", skipna=True)


# Calculate conditional probability
def calculate_conditional_probabilities(y_true, y_pred, bin_edges=np.arange(200, 280, 2)):
    bin_means, _, binnumber = binned_statistic(y_true, y_pred, statistic=np.nanmean, bins=bin_edges)
    bin_per90, _, _ = binned_statistic(y_true, y_pred, statistic=lambda x: np.nanpercentile(x, 90), bins=bin_edges)
    bin_per10, _, _ = binned_statistic(y_true, y_pred, statistic=lambda x: np.nanpercentile(x, 10), bins=bin_edges)

    h_joint_test_pred, _, _ = np.histogram2d(y_true, y_pred, bins=bin_edges, density=True)
    h_test, _ = np.histogram(y_true, bins=bin_edges, density=True)
    h_conditional = h_joint_test_pred / h_test.reshape(-1, 1)
    h_conditional_nan = np.where(h_conditional > 0, h_conditional, np.nan)
    return bin_edges, bin_means, bin_per90, bin_per10, h_test, h_conditional_nan


def plot_conditional_panel(ax, y_true, y_pred, bin_edges, title, vmin=0, vmax=0.14, show_mbe=False):
    bin_edges, bin_means, bin_per90, bin_per10, h_test, h_conditional_nan = calculate_conditional_probabilities(
        y_true=y_true, y_pred=y_pred, bin_edges=bin_edges
    )
    bin_mid = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.plot(bin_mid, bin_means, label="Mean", c="k", marker=".")
    ax.plot(bin_mid, bin_per90, label="90th percentile", c="C0", marker=".")
    ax.plot(bin_mid, bin_per10, label="10th percentile", c="C0", marker=".")
    c = ax.pcolormesh(bin_edges, bin_edges, h_conditional_nan.T, cmap="Blues", vmin=vmin, vmax=vmax)
    ax.plot([bin_mid[0], bin_mid[-1]], [bin_mid[0], bin_mid[-1]], "r--", label="True = Predicted")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.5)
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_ylim(bin_edges[0], bin_edges[-1])
    ax.set_xlabel("True [K]")
    ax.set_ylabel("Predicted [K]")
    ax.set_title(f"{title}")
    if show_mbe:
        mbe = np.nanmean(y_pred - y_true)
        ax.text(
            250,
            250,
            f"Mean Bias Error:\n{mbe:.2f} K",
            fontsize=11,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
    return c

# %% groupby latitude bins
habit_std, psd, orbits, ds_arts = load_arts_earthcare_data(0, 2)
ds_arts["cloud_top_T"] = get_cloud_top_T(ds_arts, fwc_threshold=1e-5)
plot_cloud_top_T = True  # Set to False to plot ARTS brightness temperature instead

# Group by latitude bins
groups = ds_arts.groupby_bins(
    abs(ds_arts["latitude"]),
    bins=[0, 30, 60, 90],
    labels=["low", "mid", "high"],
)

n_groups = len(groups)

fig, axes = plt.subplots(1, n_groups + 1, figsize=(6 * (n_groups + 1), 6), constrained_layout=True, sharex=True, sharey=True)
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

fig.colorbar(c, ax=axes, orientation="vertical", label="P(Predicted | True)", fraction=0.02, pad=0.04)

if plot_cloud_top_T:
    plt.figtext(
        0.5,
        -0.15,
        f"""
        This figure shows the conditional PDF of the cloud top T (predicted) given the MSI brightness temperature (true).
        Cloud top T is defined as the T at the highest height where the FWC is above a threshold (1e-5).
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
            f"../data/figures/arts_output_distribution_latitude_{habit_std}_{psd}_{len(orbits)}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f'Figure is saved to "../data/figures/arts_output_distribution_latitude_{habit_std}_{psd}_{len(orbits)}.png"')

#%%
ds_arts_groupby_orbit=ds_arts.where((ds_arts.cloud_top_T<200).compute(), drop=True).groupby('orbit_frame')
dict_group_len = {k:len(v) for k,v in ds_arts_groupby_orbit.groups.items()}
# Sort the dictionary by values in descending order and display the top 5 orbit numbers
sorted_orbits = sorted(dict_group_len.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 orbit numbers by profile count:")
for orbit, profile_count in sorted_orbits:
    print(f"Orbit: {orbit}, Profiles: {profile_count}")
    
# %% Plot total distribution of ARTS output
habit_std, psd, orbits, ds_arts = load_arts_earthcare_data(0, 0)

# Mask out low clouds
# ds_arts["cloud_top_height"] = get_cloud_top_height(ds_arts, fwc_threshold=1e-5)
ds_arts["cloud_top_T"] = get_cloud_top_T(ds_arts, fwc_threshold=1e-5)
# ds_arts_subset = ds_arts.where((ds_arts["cloud_top_T"] < 245))
ds_arts_subset = ds_arts

# Calculate mean bias error (MBE)
y_true = ds_arts_subset["pixel_values"].values.flatten()
y_pred = ds_arts_subset["arts"].mean("f_grid").values.flatten()
mbe = np.nanmean(y_pred - y_true)
bin_edges = np.arange(200, 280, 2)

fig, (ax0, ax1) = plt.subplots(2, 1, height_ratios=[3, 1], sharex=True, figsize=(7, 7), constrained_layout=True)

h_test, _ = np.histogram(y_true, bins=bin_edges, density=True)

# Plot conditional panel (main PDF)
c = plot_conditional_panel(
    ax0,
    y_true=y_true,
    y_pred=y_pred,
    bin_edges=bin_edges,
    title=f"{habit_std}\n{psd}\n{len(ds_arts_subset.nray)} profiles",
    show_mbe=True,
)

fig.colorbar(c, ax=ax0, label="P(Predicted | True)")

# Plot the distribution of the true values below
bin_mid = (bin_edges[:-1] + bin_edges[1:]) / 2
ax1.plot(bin_mid, h_test, c="C0", marker=".")
ax1.set_ylabel("P(True)")
ax1.set_xlabel("True [K]")
ax1.text(
    0.6,
    0.5,
    "MSI Tb>245 K was excluded \n from the processing",
    transform=ax1.transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
)

plt.figtext(
    0,
    -0.2,
    f"""
        This figure shows the conditional PDF of the ARTS brightness temperature (predicted) given the MSI brightness temperature (true).
        The first panel shows the mean, 90th and 10th percentiles of the predicted for each bin of true.
        The second panel shows the distribution of the true values.
        The assumed habit and PSD, and the total number of orbit frames are shown in the title.
        The total number of profiles are {len(ds_arts_subset["nray"])}.
        The ARTS brightness temperature is averaged over the f_grid (length of {len(ds_arts_subset['f_grid'])}).
    """,
    ha="left",
    fontsize=10,
    wrap=True,
)
if save:
    plt.savefig(f"../data/figures/arts_output_distribution_{habit_std}_{psd}_{len(orbits)}.png", dpi=300, bbox_inches="tight")
    print(f'Figure is saved to "../data/figures/arts_output_distribution_{habit_std}_{psd}_{len(orbits)}.png"')


# %% plot the results in time series
habit_std, psd, orbits, ds_arts = load_arts_earthcare_data(0, 0, n_files=10, random_seed=42)
ds_arts["cloud_top_T"] = get_cloud_top_T(ds_arts, fwc_threshold=1e-5)

fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 8), constrained_layout=True)

# plot the dBZ, FWC, and LWC profiles
kwargs = dict(
    y="height_grid",
    x="nray",
    add_colorbar=True,
)
# ds_arts["dBZ"].where(ds_arts["dBZ"] > -30).plot(ax=axes[0], vmin=-30, vmax=30, **kwargs)
ds_arts["temperature"].plot(ax=axes[0], vmin=200, vmax=300, **kwargs)
ds_arts["bulkprop"].sel(scat_species="FWC").pipe(np.log10).plot(ax=axes[1], vmin=-6, vmax=-3, **kwargs)
ds_arts["bulkprop"].sel(scat_species="LWC").pipe(np.log10).plot(ax=axes[2], vmin=-7, vmax=-1, **kwargs)

# for x in np.where(ds_arts.time.diff("nray").load() > pd.to_timedelta("60s"))[0]:
#     axes[0].axvline(x=x, ymin=0, ymax=1, color="r", ls="-", lw=1)

# plot the brightness temperature
kwargs = dict(x="nray", marker=".", markersize=0.3, ls="-", lw=0.3, ax=axes[3])
ds_arts["cloud_top_T"].plot(label="cloud top T", **kwargs)
ds_arts["arts"].mean("f_grid").plot(label="arts", **kwargs)
ds_arts["pixel_values"].plot(label="MSI", **kwargs)
axes[3].legend(loc="center left")

# plot the difference between ARTS and MSI brightness temperature
kwargs = dict(x="nray", marker=".", markersize=0.3, ls="-", lw=0.3, ax=axes[4], ylim=[-55, 55])
ds_arts[["pixel_values", "cloud_top_T"]].to_array().diff("variable").assign_attrs(units="K").plot(label="cloud top T", **kwargs)
ds_arts[["pixel_values", "arts"]].to_array().diff("variable").mean("f_grid").assign_attrs(units="K").plot(label="arts", **kwargs)
axes[4].axhline(0, color="k", ls="--", lw=0.5)
axes[4].legend(loc="center left")

axes[0].set_title("Temperature")
axes[1].set_title("FWC (log10)")
axes[2].set_title("LWC (log10)")
axes[3].set_title("Brightness Temperature (mean over f_grid)")
axes[4].set_title("Diff (X - MSI)")
axes[0].set_xlabel("")
axes[1].set_xlabel("")
axes[2].set_xlabel("")
axes[3].set_xlabel("")

fig.suptitle(
    f"""
    {habit_std}
    {psd}
    {len(orbits)} orbit frames
    """,
)
# put some text under the figure
plt.figtext(
    0.5,
    -0.1,
    """
    This figure shows the results of the ARTS simulation for EarthCARE data.
    The first three panels show the T, FWC (log10), and LWC (log10) profiles.
    The fourth panel shows the brightness temperature from ARTS and MSI.
    The last panel shows the difference between ARTS and MSI brightness temperature.
    The cloud top T is defined as the T at the highest height where the FWC is above a threshold (1e-5).
""",
    ha="center",
    fontsize=10,
    wrap=True,
)
# plt.show()
if save:
    plt.savefig(f"../data/figures/arts_output_{habit_std}_{psd}_{len(orbits)}.png", dpi=300, bbox_inches="tight")


# %% read an input file
# path = f"../data/earthcare/arts_input_data/arts_input_{'03645E'}.nc"
# ds = xr.open_dataset(path)

# fig, ax = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True, sharex=True)
# ds.dBZ.plot(ax=ax[0], x="nray", y="height_grid", vmin=-30, vmax=30, add_colorbar=True)
# ds.pixel_values.plot(ax=ax[1], x="nray", marker=".", markersize=1, label="MSI")
