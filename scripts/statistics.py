# %%
from plotting import (
    load_arts_output_data,
    plot_conditional_panel,
    calculate_conditional_probabilities,
    sci_formatter,
)
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
from earthcare_ir import habit_std_list, psd_list, get_cloud_top_height, get_cloud_top_T
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator, LogFormatter
import xgboost as xgb

# ignore invalid value warnings
import warnings

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered in divide *"
)
save = False

file_pattern = (
    "../data/earthcare/arts_output_data/high_fwp_5th_{habit_std}_{psd}_{orbit_frame}.nc"
)


# %% load all habit and/or psd arts data
data = []
for j in range(len(psd_list)):
    for i in range(len(habit_std_list)):
        data.append(
            load_arts_output_data(
                i,
                j,
                file_pattern=file_pattern,
                n_files=100,  # test on small amount first
            )
        )

# %% plot FWC distribution
fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ls = ["-", "--", ":"]
c = ["#0071B2FF", "#D55C00FF", "#009E74FF"]
for j in range(len(psd_list)):
    for i in range(len(habit_std_list)):
        psd = psd_list[j]
        habit_std = habit_std_list[i]
        ds_arts = [d[-1] for d in data if d[0] == habit_std and d[1] == psd][0]

        bins_fwc = np.logspace(-6, -2, 50)
        hist_fwc, _ = np.histogram(
            ds_arts.frozen_water_content,
            bins=bins_fwc,
            density=True,
        )
        ax.plot(
            bins_fwc[:-1],
            hist_fwc,
            label=f"{psd}",
            ls=ls[i],
            color=c[j],
            alpha=0.8,
        )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([1e3, 1e5])
ax.set_xlabel("Frozen Water Content [kg/m³]")
ax.set_ylabel("Probability Density")
# ax.legend(loc="lower left", framealpha=0.3)
ax.set_title("FWC Distribution")

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    np.array(handles)[::3], np.array(labels)[::3], loc="lower left", framealpha=0.3
)

if save:
    plt.savefig(
        f"../data/figures/arts_output_fwc_distribution.png",
        dpi=1000,
        bbox_inches="tight",
    )
    print(f'Figure is saved to "../data/figures/arts_output_fwc_distribution.png"')

# %% calculate bin statistics and conditional probabilities
bin_means_, bin_per90_, bin_per10_, h_conditional_nan_ = [], [], [], []
bin_edges = np.arange(180, 300, 5)
for i in range(len(data)):
    _, bin_means, bin_per90, bin_per10, _, h_conditional_nan = (
        calculate_conditional_probabilities(
            y_true=data[i][-1]["pixel_values"].values.flatten(),
            y_pred=data[i][-1]["arts"].mean("f_grid").values.flatten(),
            bin_edges=bin_edges,
        )
    )
    bin_means_.append(bin_means)
    bin_per10_.append(bin_per10)
    bin_per90_.append(bin_per90)
    h_conditional_nan_.append(h_conditional_nan)

    habit = data[i][0]
    psd = data[i][1]

    if save:
        np.save(
            f"../data/earthcare/arts_output_statistics/conditional_probabilities_{psd}_{habit}.npy",
            {
                "bin_edges": bin_edges,
                "bin_means": bin_means,
                "bin_per90": bin_per90,
                "bin_per10": bin_per10,
                "h_conditional_nan": h_conditional_nan,
            },
        )

save = False

# %% Plot total distribution of ARTS output
# load all pre caculated statistics
bin_edges_, bin_means_, bin_per90_, bin_per10_, h_conditional_nan_ = [], [], [], [], []
psds, habits = [], []
for i in range(len(habit_std_list)):
    for j in range(len(psd_list)):
        psd, habit = psd_list[j], habit_std_list[i]
        print(psd, habit)
        psds.append(psd)
        habits.append(habit)
        stats_dict = np.load(
            f"/home/anqil/arts_ir_simulation/data/earthcare/arts_output_statistics/conditional_probabilities_{psd}_{habit}.npy",
            allow_pickle=True,
        ).item()
        bin_edges_.append(stats_dict["bin_edges"])
        bin_means_.append(stats_dict["bin_means"])
        bin_per90_.append(stats_dict["bin_per90"])
        bin_per10_.append(stats_dict["bin_per10"])
        h_conditional_nan_.append(stats_dict["h_conditional_nan"])


# plot the statitstics
def plot_conditional_panel(
    ax,
    bin_edges,
    bin_means,
    bin_per90,
    bin_per10,
    h_conditional_nan,
    title,
    norm,
):

    bin_mid = (bin_edges[:-1] + bin_edges[1:]) / 2
    (l_mean,) = ax.plot(bin_mid, bin_means, label="Mean", c="lightgrey", ls="-", lw=3)
    (l_per90,) = ax.plot(
        bin_mid, bin_per90, label="90th percentile", c="lightgrey", ls=":", lw=2
    )
    (l_per10,) = ax.plot(
        bin_mid, bin_per10, label="10th percentile", c="lightgrey", ls=":", lw=2
    )
    c = ax.pcolormesh(
        bin_edges, bin_edges, h_conditional_nan.T, cmap="Blues", norm=norm
    )
    ax.plot(
        [bin_mid[0], bin_mid[-1]],
        [bin_mid[0], bin_mid[-1]],
        "r--",
        label="True = Pred",
    )
    ax.legend(
        [l_mean, l_per90],
        ["Mean", "P10/P90"],
        loc="lower right",
        fontsize=8,
        framealpha=1,
        # facecolor="lightgrey",
    )
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_ylim(bin_edges[0], bin_edges[-1])
    ax.set_title(title)

    return ax

bin_edges = bin_edges_[0]
vmin = max(1e-6, np.nanmin(h_conditional_nan_))
vmax = np.nanmax(h_conditional_nan_)
norm = LogNorm(vmin=vmin, vmax=vmax)
fig, ax = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True, constrained_layout=True)
for i in range(len(h_conditional_nan_)):
    psd = psds[i].replace("Exponential", "Mod.Gamma")
    habit = habits[i]
    plot_conditional_panel(
        ax.flatten()[i],
        bin_edges,
        bin_means_[i],
        bin_per90_[i],
        bin_per10_[i],
        h_conditional_nan_[i],
        norm=norm,
        title=f"{psd}\n{habit}",
    )
    ax.flatten()[i].legend().set_visible(False)

    if i in [0, 3, 6]:
        ax.flatten()[i].set_ylabel("Predicted [K]")
    if i in [6, 7, 8]:
        ax.flatten()[i].set_xlabel("True [K]")
cbar = plt.colorbar(
    ax.flatten()[0].collections[0],
    ax=ax,
    orientation="vertical",
    label="P (Predicted | True)",
    aspect=40,
)
cbar.locator = LogLocator(base=10)
cbar.update_ticks()
cbar.formatter = FuncFormatter(sci_formatter)

# %% groupby latitude bins
# habit_std, psd, orbits, ds_arts = load_arts_output_data(2, 2, file_pattern=file_pattern)
# ds_arts = get_cloud_top_T(ds_arts, fwc_threshold=1e-5)
# plot_cloud_top_T = False  # Set to False to plot ARTS brightness temperature instead

# # Group by latitude bins
# groups = ds_arts.groupby_bins(
#     abs(ds_arts["latitude"]),
#     bins=[0, 30, 60, 90],
#     labels=["low", "mid", "high"],
# )

# n_groups = len(groups)

# fig, axes = plt.subplots(
#     1,
#     n_groups + 1,
#     figsize=(6 * (n_groups + 1), 6),
#     constrained_layout=True,
#     sharex=True,
#     sharey=True,
# )
# if n_groups + 1 == 1:
#     axes = np.array([axes])

# vmin, vmax = 0, 0.14
# bin_edges_global = np.arange(180, 280, 2)


# # Total (all data)
# y_true_total = ds_arts["pixel_values"].values.flatten()
# if plot_cloud_top_T:
#     y_pred_total = ds_arts["cloud_top_T"].values.flatten()
# else:
#     y_pred_total = ds_arts["arts"].mean("f_grid").values.flatten()

# c = plot_conditional_panel(
#     ax=axes[0],
#     y_true=y_true_total,
#     y_pred=y_pred_total,
#     bin_edges=bin_edges_global,
#     title=f"Total\n{habit_std}, {psd}, {len(ds_arts.nray)} profiles",
#     vmin=vmin,
#     vmax=vmax,
#     show_mbe=True,
# )

# # Latitude groups
# for i, (group_name, ds_group) in enumerate(groups):
#     y_true_group = ds_group["pixel_values"].values.flatten()
#     if plot_cloud_top_T:
#         y_pred_group = ds_group["cloud_top_T"].values.flatten()
#     else:
#         y_pred_group = ds_group["arts"].mean("f_grid").values.flatten()

#     plot_conditional_panel(
#         ax=axes[i + 1],
#         y_true=y_true_group,
#         y_pred=y_pred_group,
#         bin_edges=bin_edges_global,
#         title=f"{group_name.capitalize()} latitude\n{habit_std}, {psd}, {len(ds_group.nray)} profiles",
#         vmin=vmin,
#         vmax=vmax,
#         show_mbe=True,
#     )

# fig.colorbar(
#     c,
#     ax=axes,
#     orientation="vertical",
#     label="P(Predicted | True)",
#     fraction=0.02,
#     pad=0.04,
# )

# if plot_cloud_top_T:
#     plt.figtext(
#         0.5,
#         -0.15,
#         f"""
#         This figure shows the conditional PDF of the cloud top T (predicted) given the MSI brightness temperature (true).
#         Cloud top T is defined as the T at the highest height where the FWC is above a threshold ({ds_arts['cloud_top_T'].attrs['fwc_treshhold']}).
#         """,
#         ha="center",
#         fontsize=10,
#         wrap=True,
#     )

# else:
#     plt.figtext(
#         0.5,
#         -0.15,
#         f"""
#         This figure shows the conditional PDF of the ARTS brightness temperature (predicted) given the MSI brightness temperature (true).
#         The first panel shows the total distribution, while the other panels show the distribution for different latitude bins.
#         The assumed habit and PSD, and the total number of orbit frames are shown in the title.
#         The ARTS brightness temperature is averaged over the f_grid (length of {len(ds_arts['f_grid'])}).
#         """,
#         ha="center",
#         fontsize=10,
#         wrap=True,
#     )

#     if save:
#         plt.savefig(
#             f"../data/figures/arts_output_distribution_latitude_{habit_std}_{psd}.png",
#             dpi=300,
#             bbox_inches="tight",
#         )
#         print(f'Figure is saved to "../data/figures/arts_output_distribution_latitude_{habit_std}_{psd}.png"')
