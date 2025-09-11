# %%
import string
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator, LogFormatter
from earthcare_ir import habit_std_list, psd_list

from pyarts.workspace import Workspace
import pyarts.xml as xml
import os
import xgboost as xgb
import xarray as xr
from plotting import load_arts_output_data, sci_formatter

save = False
file_pattern = "../data/earthcare/arts_output_data/high_fwp_5th_{habit_std}_{psd}_{orbit_frame}.nc"

# %%
def load_ml_model_and_predict(ds_arts):
    # Load ML model
    model_tag = "all_orbits_20250101_20250501_seed42"
    model = xgb.Booster()
    model.load_model(f"/home/anqil/earthcare/data/xgb_regressor_{model_tag}.json")
    print("Model loaded from file")

    # construct training data and make prediction
    orbit_frame = ds_arts.orbit_frame.data.item()
    path_to_data = f"/home/anqil/earthcare/data/training_data/training_data_{orbit_frame}.nc"
    ds = xr.open_dataset(path_to_data)
    ds = ds.set_xindex("time").sel(time=ds_arts.time)
    y_pred_ml = model.predict(xgb.DMatrix(ds.x))[:, 1]  # select only channel 1
    return y_pred_ml

def plot_conditional_panel(
    ax,
    bin_edges,
    bin_means,
    bin_per90,
    bin_per10,
    h_conditional_nan,
    title,
    norm,
    show_mbe=False,
    mbe=None,
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
    ax.set_xlabel("True [K]")
    ax.set_ylabel("Predicted [K]")
    ax.set_title(f"{title}")
    if show_mbe:
        if mbe is not None:
            pass
        else:
            raise ValueError("mbe must be provided if show_mbe is True")
        ax.text(
            250,
            250,
            f"Mean Bias Error:\n{mbe:.2f} K",
            fontsize=11,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
    return c

def load_ml_statistics():
    model_tag = "all_orbits_20250101_20250501_seed42"
    data_dict = np.load(
        f"/home/anqil/earthcare/data/eval_results_{model_tag}/cond_prob/conditional_probabilities.npy",
        allow_pickle=True,
    ).item()
    h_conditional_nan_ = data_dict["density"]
    bin_means_ = data_dict["bin_means"]
    bin_per90_ = data_dict["bin_per90"]
    bin_per10_ = data_dict["bin_per10"]
    bin_edges = data_dict["bin_edges"]
    return bin_edges, bin_means_, bin_per90_, bin_per10_, h_conditional_nan_


def label_line(
    ax,
    line,
    label=None,
    x=None,
    x_frac=None,
    fontsize=8,
    pad=4,
    rotate=True,
    **text_kwargs,
):
    """
    Place `label` on `line` at x (data coordinate) or at x_frac (fraction of x-range).
    If label is None, use line.get_label(); ignores lines with '_nolegend_'.

    If rotate=True the label will be rotated to match the local slope of the
    line in display (pixel) coordinates so the rotation looks correct on
    log-scaled axes as well.
    """
    if label is None:
        label = line.get_label()
        if label.startswith("_nolegend_"):
            return

    xd = np.asarray(line.get_xdata())
    yd = np.asarray(line.get_ydata())
    if x_frac is not None:
        x = xd.min() + x_frac * (xd.max() - xd.min())
    if x is None:
        # default: mid-point
        x = xd.mean()
    # clamp x to data range
    x = np.clip(x, xd.min(), xd.max())
    y = np.interp(x, xd, yd)

    color = text_kwargs.pop("color", line.get_color())

    rotation = 0.0
    if rotate:
        # compute a small offset in data coordinates for slope estimation
        dx = (xd.max() - xd.min()) * 1e-3
        # ensure we stay inside data range
        if x + dx <= xd.max():
            x2 = x + dx
        else:
            x2 = x - dx
        y2 = np.interp(x2, xd, yd)
        # transform to display coordinates so angle accounts for scales/transforms
        p1 = ax.transData.transform((x, y))
        p2 = ax.transData.transform((x2, y2))
        dx_disp = p2[0] - p1[0]
        dy_disp = p2[1] - p1[1]
        rotation = np.degrees(np.arctan2(dy_disp, dx_disp))

    # place label, centered on the evaluated point and rotated to match slope
    ax.text(
        x,
        y,
        label,
        fontsize=fontsize,
        color=color,
        rotation=rotation,
        rotation_mode="anchor",
        va=text_kwargs.pop("va", "bottom"),
        ha=text_kwargs.pop("ha", "center"),
        bbox=text_kwargs.pop(
            "bbox",
            dict(facecolor="none", edgecolor="none", pad=pad),
        ),
        clip_on=True,
        **text_kwargs,
    )


def set_psd(ws, psd):
    if psd == psd_list[0]:
        ws.psdDelanoeEtAl14(
            t_max=275,
            t_min=180,
            n0Star=-999,
            Dm=-999,
            alpha=-0.237,
            beta=1.839,
        )
    elif psd == psd_list[1]:
        ws.psdFieldEtAl07(
            scat_species_a=0.02,
            scat_species_b=2,
            regime="TR",
            # For low and high temperatures some parameters with default
            # values could matter
        )
    elif psd == psd_list[2]:
        ws.psdModifiedGammaMass(
            scat_species_a=0.02,
            scat_species_b=2,
            n0=1e10,
            ga=1.5,
            mu=0,
            la=-999,
            t_max=373,
            t_min=0,
        )


# %% load arts ouput statistics
bin_edges_, bin_means_, bin_per90_, bin_per10_, h_conditional_nan_ = [], [], [], [], []
psds, habits = [], []
for i, j in [(0, 0), (0, 2), (1, 0)]:
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


# %% Plot ARTS and ML distribution
# TODO: check if bin_edges are consistent
bin_edges = bin_edges_[0]
vmin = max(1e-6, np.nanmin(h_conditional_nan_))
vmax = np.nanmax(h_conditional_nan_)
norm = LogNorm(vmin=vmin, vmax=vmax)
fig, ax = plt.subplots(
    2, 2, figsize=(6, 6), sharex=True, sharey=True, constrained_layout=True
)
for i in range(3):
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

    if i in [1, 3]:
        ax.flatten()[i].set_ylabel("")
    if i in [0, 1]:
        ax.flatten()[i].set_xlabel("")

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

# add ML distribution
statistics_ml = load_ml_statistics()
if ~(statistics_ml[0] == bin_edges).all():
    print("bin_edges are inconsistent")
else:
    print("bin_edges are consistent")
    band_idx = 1
    _, bin_means_ml, bin_per90_ml, bin_per10_ml, h_conditional_nan_ml = statistics_ml

    plot_conditional_panel(
        ax.flatten()[-1],
        bin_edges,
        bin_means_ml[band_idx],
        bin_per90_ml[band_idx],
        bin_per10_ml[band_idx],
        h_conditional_nan_ml[band_idx],
        norm=norm,
        title=f"ML XGBoost",
    )
    ax.flatten()[-1].set_ylabel("")

# add 'a)' 'b)'  labels
labels = [f"{l})" for l in string.ascii_lowercase]  # ['a)', 'b)', ...]
for ax, lab in zip(ax.flatten(), labels):
    ax.text(
        0.02,
        0.98,
        lab,  # small offset from upper-left in axes coords
        transform=ax.transAxes,  # use axes coordinate system (0..1)
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="none", alpha=0.6, edgecolor="none"),
    )

if save:
    fig.savefig(
        f"../data/figures/arts_output_cond_prob_w_ml.png", dpi=1000, bbox_inches="tight"
    )
    print(f'Figure is saved to "../data/figures/arts_output_cond_prob_w_ml.png"')
    save = False

# %%
fig, ax_extra = plt.subplots(1, 1, figsize=(2.5, 3.5), constrained_layout=True)
# % Add extra plot of PSD
# ax_extra = fig.add_axes([0.92, 0.08, 0.3, 0.365], transform=fig.transFigure)

datafolder = "/data/s5/scattering_data/IRDbase/Yang2016/ArtsFormat"
habit = "8-ColumnAggregate-Smooth"
M = xml.load(os.path.join(datafolder, f"{habit}.meta.xml"), search_arts_path=False)
d_veq_Yang2016 = [getattr(M[i], "diameter_volume_equ", None) for i in range(len(M))]

ws = Workspace(verbosity=0)
ws.dpnd_data_dx_names = []
# ws.psd_size_grid = np.logspace(-5, -2, 21)
ws.psd_size_grid = np.array(d_veq_Yang2016)[::4]  # Use sizes from Yang2016
ws.pnd_agenda_input_t = np.array([210, 230, 260])  # Temperature
iwc_fixed = 1e-4  # Fixed IWC value in kg/m3
ws.pnd_agenda_input = iwc_fixed * np.ones(
    [len(ws.pnd_agenda_input_t.value), 1]
)  # IWC in kg/m3
ws.pnd_agenda_input_names = ["FWC"]  # Name does not matter, just order!

line_styles = ["-", "--", ":"]
c = ["#0071B2FF", "#D55C00FF", "#009E74FF"]
for i, psd in enumerate(psd_list):
    set_psd(ws, psd)
    # Plot each column (temperature) separately with different line styles but same color
    for j, temp in enumerate(ws.pnd_agenda_input_t.value):
        (l,) = ax_extra.plot(
            ws.psd_size_grid.value,
            ws.psd_data.value[j, :],
            ls=line_styles[j],
            color=c[i],
            alpha=0.8,
            label=f"{psd.replace('Exponential', 'MG').replace('DelanoeEtAl14', 'D14').replace('FieldEtAl07TR', 'F07TR')}, {temp:.0f}K",
        )
ax_extra.set_yscale("log")
ax_extra.set_xscale("log")
ax_extra.set_ylim([1e6, 1e11])
ax_extra.set_xlim([1e-6, 2e-3])
ax_extra.set_title(f"FWC = {sci_formatter(iwc_fixed, None)} $kg/m^3$",fontsize=10)
ax_extra.set_xlabel("Diameter Volume-eq. [m]")

for i, l, x in zip(
    [0, 1, 2, 3, 4, 5, 6],
    [
        "210K",
        "230K",
        "D14 260K",
        "F07 210K",
        "230K",
        "260K",
        "   M.Gamma",
    ],
    [2e-6, 2e-6, 1.5e-6, 2e-6, 1e-6, 2e-5, 4e-5],
):
    label_line(ax_extra, ax_extra.lines[i], label=l, x=x, ha="left", va="bottom", fontsize=9)

if save:
    fig.savefig(
        f"../data/figures/psd_comparison_fixedIWC_{iwc_fixed}.png",
        dpi=1000,
        bbox_inches="tight",
        transparent=True,
    )
    print(
        f'Figure is saved to "../data/figures/psd_comparison_fixedIWC_{iwc_fixed}.png"'
    )
    save = False


#%% create sine wave
# color = "#B23200FF"
color = "#0083B2FF"
x = np.linspace(0, 8 * np.pi, 400)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.plot(x, y, lw=8, color=color)

# hide axes
ax.set_axis_off()

# make the axes occupy the full figure (no whitespace)
ax.set_position([0, 0, 1, 1])

# ensure figure background is transparent
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# save cropped, transparent PNG
fig.savefig(
    "../data/figures/sine_full_bleed_mv.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
    transparent=True,
)


# %% plot one orbit in time series
orbit_frame = '04549D' #"04015F"#'04701A'#'03871C'#"04015F"

habits = []
psds = []
arts_T = []
arts_top_height = []
# Loop through all habit_std and psd combinations
i = 2  # habit index
for j in range(3):  # psd index
    habit_std, psd, orbits, ds_arts = load_arts_output_data(i, j, orbit_frame=orbit_frame, file_pattern=file_pattern)
    # ds_arts = get_cloud_top_height(ds_arts, fwc_threshold=2e-5)
    habits.append(habit_std)
    psds.append(psd)
    arts_T.append(ds_arts["arts"].mean("f_grid"))
    # arts_top_height.append(ds_arts["cloud_top_height"])

ds_arts = ds_arts.assign(y_pred_ml=("nray", load_ml_model_and_predict(ds_arts)))
ds_arts["height_grid"] = ds_arts["height_grid"] * 1e-3
psds = [p.replace("Exponential", "M.Gamma") for p in psds]

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 5), constrained_layout=True)

# plot the dB profiles
kwargs = dict(
    y="height_grid",
    x="nray",
    add_colorbar=True,
)
ds_arts["dBZ"].where(ds_arts["dBZ"] > -30).plot(ax=axes[0], vmin=-30, vmax=30, **kwargs)

colors = ["#0071B2FF", "#D55C00FF", "#009E74FF"]
# plot the brightness temperature
kwargs = dict(ax=axes[1], x="nray", ls="-", lw=1, ylim=[200, 280])
[a.plot(label=f"{p}", c=c, **kwargs) for c, p, a in zip(colors, psds, arts_T)]
ds_arts["y_pred_ml"].plot(label="XGBoost", c="r", **kwargs)
ds_arts["pixel_values"].plot(label="MSI", c="grey", **kwargs)

# plot the difference between ARTS and MSI brightness temperature
kwargs = dict(ax=axes[2], x="nray", ls="-", lw=1, ylim=[-25, 25])
[(a - ds_arts["pixel_values"]).assign_attrs(units="K").plot(label=f"{p}", c=c, **kwargs) for c, p, a in zip(colors, psds, arts_T)]
(ds_arts["y_pred_ml"] - ds_arts["pixel_values"]).plot(label="XGBoost", c="r", **kwargs)
axes[2].axhline(0, color="k", ls="--", lw=1)

handles, labels = axes[1].get_legend_handles_labels()
labels_cust = ['D14', 'F07', 'M.Gamma', 'XGBoost', 'MSI']  # custom labels
fig.legend(handles, labels_cust, loc="center right", bbox_to_anchor=(1.05, 0.525), framealpha=0.3)
# optionally remove the axes legend:
axes[1].legend().set_visible(False)

axes[0].set_title(f"Radar Reflectivity (CPR)")
axes[1].set_title("IR Temperature")
axes[2].set_title("Predicted - True (MSI)")
axes[0].set_xlabel("")
axes[1].set_xlabel("")
axes[2].set_xlabel("Profile number")
axes[0].set_ylabel("z [km]")
axes[1].set_ylabel("Tb [K]")
axes[2].set_ylabel("Diff [K]")

# add 'a)' 'b)'  labels
labels = [f"{l})" for l in string.ascii_lowercase]  # ['a)', 'b)', ...]
for ax, lab in zip(axes.flatten(), labels):
    ax.text(
        0.02,
        1.2,
        lab,  # small offset from upper-left in axes coords
        transform=ax.transAxes,  # use axes coordinate system (0..1)
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="none", alpha=0.6, edgecolor="none"),
    )

if save:
    fig.savefig(
        f"../data/figures/arts_output_w_ml_series_{orbit_frame}.png",
        dpi=1000,
        bbox_inches="tight",
    )
    print(f'Figure is saved to "../data/figures/arts_output_w_ml_series_{orbit_frame}.png"')
    save = False
