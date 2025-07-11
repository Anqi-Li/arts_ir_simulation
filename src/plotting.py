# %%
from cycler import cycler
from ir_earthcare import get_cloud_top_T
from pyarts.workspace import Workspace
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from ir_earthcare import habit_std_list, psd_list
import xarray as xr
import glob

# ignore warnings from divide by zero encountered in log10
import warnings

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="divide by zero encountered in log10"
)


# %%
def plot_psd_mgd(
    coefs_mgd: dict,
    psd_size_grid=np.linspace(1e-6, 50e-6),
    y_scale="log",
    x_scale="linear",
):
    ws = Workspace(verbosity=0)
    ws.psd_size_grid = psd_size_grid
    ws.dpnd_data_dx_names = []

    # Setting all 4 parameters by GIN
    # This generates a single PSD
    ws.pnd_agenda_input_t = np.array([190])  # Not really used
    ws.pnd_agenda_input = np.array([[]])
    ws.pnd_agenda_input_names = []
    ws.psdModifiedGamma(
        n0=coefs_mgd.get("n0"),
        ga=coefs_mgd.get("ga"),
        la=coefs_mgd.get("la", 1e7),  # Default value if not provided
        mu=coefs_mgd.get("mu", 0),  # Default value if not provided
        t_min=0,
        t_max=999,
        picky=1,
    )
    plt.figure()
    plt.plot(
        ws.psd_size_grid.value * 1e6,
        ws.psd_data.value[0],
        label=f"N₀={coefs_mgd.get('n0'):.1e}, μ={coefs_mgd.get('mu', 0)}, λ={coefs_mgd.get('la', 1e7):.1e}, γ={coefs_mgd.get('ga')}",
        c="C0",
    )
    plt.yscale(y_scale)
    plt.xscale(x_scale)
    plt.xlabel("Diameter (µm)")
    plt.ylabel("PSD (µm⁻¹ cm⁻³)")
    plt.legend()
    plt.grid()
    plt.show()


def plot_fwc_and_temperatures(
    ds,
    fwc_threshold=None,
    temperature_vars=["pixel_values", "cloud_top_T"],
    add_diff=False,
):
    if fwc_threshold is None and "cloud_top_T" not in ds:
        raise ValueError(
            "Please provide a fwc_threshold or ensure 'cloud_top_T' is in the dataset."
        )
    if fwc_threshold is not None:
        ds = get_cloud_top_T(ds, fwc_threshold=fwc_threshold)

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 6), constrained_layout=True)
    ds["frozen_water_content"].pipe(np.log10).plot(
        ax=ax[0],
        x="nray",
        y="height_grid",
        cmap="viridis",
        cbar_kwargs={"label": "Frozen Water Content log10(kg/m^3)"},
    )
    if "cloud_top_height" in ds:
        ds["cloud_top_height"].plot(
            ax=ax[0],
            x="nray",
            label="Cloud Top Height",
            color="orange",
        )

    ds[temperature_vars].to_array().plot.line(
        ax=ax[1],
        x="nray",
        hue="variable",
        add_legend=True,
    )
    if add_diff:
        ax1_twinx = ax[1].twinx()
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        new_color_cycle = cycler(color=default_colors[1:])
        ax1_twinx.set_prop_cycle(new_color_cycle)

        ds[temperature_vars].to_array("diff").diff("diff").plot.line(
            ax=ax1_twinx,
            x="nray",
            hue="diff",
            add_legend=True,
            linestyle="--",
            alpha=0.5,
        )
        ax1_twinx.set_ylabel("Temperature Difference (K)")

    ax[0].set_ylabel("Height (m)")
    ax[0].set_title("Frozen Water Content")
    ax[0].set_xlabel("")
    ax[0].legend()
    ax[1].set_xlabel("Ray Index")
    ax[1].set_ylabel("Temperature (K)")
    ax[1].set_title("Temperatures")

    return fig, ax


# Calculate conditional probability
def calculate_conditional_probabilities(
    y_true, y_pred, bin_edges=np.arange(180, 280, 2)
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
    return bin_edges, bin_means, bin_per90, bin_per10, h_test, h_conditional_nan


def plot_conditional_panel(
    ax, y_true, y_pred, bin_edges, title, vmin=0, vmax=0.14, show_mbe=False
):
    bin_edges, bin_means, bin_per90, bin_per10, h_test, h_conditional_nan = (
        calculate_conditional_probabilities(
            y_true=y_true, y_pred=y_pred, bin_edges=bin_edges
        )
    )
    bin_mid = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.plot(bin_mid, bin_means, label="Mean", c="k", marker=".")
    ax.plot(bin_mid, bin_per90, label="90th percentile", c="C0", marker=".")
    ax.plot(bin_mid, bin_per10, label="10th percentile", c="C0", marker=".")
    c = ax.pcolormesh(
        bin_edges, bin_edges, h_conditional_nan.T, cmap="Blues", vmin=vmin, vmax=vmax
    )
    ax.plot(
        [bin_mid[0], bin_mid[-1]],
        [bin_mid[0], bin_mid[-1]],
        "r--",
        label="True = Predicted",
    )
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


# %% read output files
def load_arts_output_data(
    habit_std_idx, psd_idx, orbit_frame=None, n_files=None, random_seed=42
):
    habit_std = habit_std_list[habit_std_idx]
    psd = psd_list[psd_idx]
    print(f"Habit: {habit_std}, PSD: {psd}")
    if orbit_frame is None:
        orbit_frame = "*"  # Use wildcard to match all orbit frames

    pattern = (
        f"../data/earthcare/arts_output_data/high_fwp_5th_{habit_std}_{psd}_{orbit_frame}.nc"
        # f'/scratch/li/arts-earthcare/arts_output_data_old/cold_allsky_{habit_std}_{psd}_{orbit_frame}.nc'
    )
    matching_files = sorted(glob.glob(pattern))
    if n_files is not None and len(matching_files) > n_files:
        rng = np.random.default_rng(random_seed)
        matching_files = rng.choice(matching_files, size=n_files, replace=False)
        matching_files = matching_files.tolist()
    orbits = [o[-9:-3] for o in matching_files]
    print(f"Number of matching files: {len(matching_files)}")

    datasets = [
        xr.open_dataset(f, chunks="auto").assign_coords(orbit_frame=f[-9:-3])
        for f in matching_files
    ]
    ds_arts = xr.concat(datasets, dim="nray").sortby("profileTime")

    return habit_std, psd, orbits, ds_arts


def plot_arts_output_distribution(habit_std_idx, psd_idx, save=False, save_path=None):
    habit_std, psd, orbits, ds_arts = load_arts_output_data(habit_std_idx, psd_idx)

    # Mask out low clouds
    ds_arts = get_cloud_top_T(ds_arts, fwc_threshold=5e-5)
    ds_arts_subset = ds_arts

    # Calculate mean bias error (MBE)
    y_true = ds_arts_subset["pixel_values"].values.flatten()
    y_pred = ds_arts_subset["arts"].mean("f_grid").values.flatten()
    # y_pred = ds_arts_subset["cloud_top_T"].values.flatten()
    mbe = np.nanmean(y_pred - y_true)
    bin_edges = np.arange(180, 280, 2)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, height_ratios=[3, 1], sharex=True, figsize=(7, 7), constrained_layout=True
    )

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
        if save_path is None:
            save_path = (
                f"../data/figures/arts_output_distribution_{habit_std}_{psd}.png"
            )
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
        print(f'Figure is saved to "{save_path}"')
