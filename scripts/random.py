# %%
from onion_table import get_ds_onion_invtable
from plotting import load_arts_output_data, load_ml_model_and_predict
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from earthcare_ir import get_cloud_top_height, get_cloud_top_T, habit_std_list, psd_list
from pyarts.workspace import Workspace
from cycler import cycler

file_pattern = (
    "../data/earthcare/arts_output_data/high_fwp_5th_{habit_std}_{psd}_{orbit_frame}.nc"
)
# %%

ds_onion_invtables = np.empty((3, 3), dtype=object)
fig, ax = plt.subplots(3, 3, figsize=(8, 8), sharey=True, sharex=True)
for i in range(3):
    for j in range(3):
        habit_std = habit_std_list[i]
        psd = psd_list[j]
        print(f"Loading invtable for {habit_std}, {psd}...")

        if psd == "Exponential":
            # coefficients for the exponential PSD
            coef_mgd = {
                "n0": 1e10,  # Number concentration
                "ga": 1.5,  # Gamma parameter
            }
        else:
            coef_mgd = None
        ds_onion_invtable = get_ds_onion_invtable(
            habit=habit_std,
            psd=psd,
            coef_mgd=coef_mgd,
        ).sel(radiative_properties="FWC")
        ds_onion_invtables[i, j] = ds_onion_invtable
        da_invonion = ds_onion_invtable.sel(
            dBZ=[-30, -25, -20],
            # Temperature=slice(195, 210),
        )
        da_invonion.plot(ax=ax[i, j], x="Temperature", hue="dBZ", marker=".")

        ax[i, j].set_title(f"{da_invonion.attrs['habit']}\n{da_invonion.attrs['psd']}")
        if i == 2:
            ax[i, j].set_xlabel("Temperature (K)")
        else:
            ax[i, j].set_xlabel("")
        if j == 0:
            ax[i, j].set_ylabel("log10(FWC (kg m-3))")
        else:

            ax[i, j].set_ylabel("")
        ax[i, j].grid()

fig.tight_layout()


# %% plot psd
def set_psd(ws, psd):
    if psd == "DelanoeEtAl14":
        ws.psdDelanoeEtAl14(
            t_max=275,
            t_min=180,
            n0Star=-999,
            Dm=-999,
            alpha=-0.237,
            beta=1.839,
        )
    elif psd == "FieldEtAl07TR":
        ws.psdFieldEtAl07(
            scat_species_a=0.02,
            scat_species_b=2,
            regime="TR",
            # For low and high temperatures some parameters with default
            # values could matter
        )
    elif psd == "Exponential":
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


def get_psd(fwc=None, t=None, psd=None, psd_size_grid=np.logspace(-6, -2, 100)):
    if len(fwc) != len(t):
        raise ValueError("fwc and t must have the same length")
    ws = Workspace(verbosity=0)
    ws.dpnd_data_dx_names = []
    ws.psd_size_grid = psd_size_grid  # Particle size grid in meters
    ws.pnd_agenda_input_t = t  # Temperature
    ws.pnd_agenda_input = fwc.reshape(-1, 1)  # FWC in kg/m3
    ws.pnd_agenda_input_names = ["FWC"]  # Name does not matter, just order!
    set_psd(ws, psd)
    return ws.psd_size_grid.value, ws.psd_data.value


fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
# colors = ["b",  "r", "g","c", "m"]
colors = ["#0071B2FF", "#D55C00FF", "#009E74FF"]
linestyles = ["-", "--", ":", "-."]
for j in range(len(psd_list)):
    
    habit_idx = 0  # habit index
    dBZ = -20
    da_invonion_sel = ds_onion_invtables[habit_idx, j].sel(
        dBZ=dBZ, Temperature=[200, 220, 240]
    )
    psd = da_invonion_sel.attrs["psd"]
    fwc = da_invonion_sel.pipe(lambda x: 10**x).data
    psd_size_grid, psd_data = get_psd(
        fwc=fwc,
        t=da_invonion_sel.Temperature.data,
        psd=psd,
    )
    psd_label = (
        psd.replace("Exponential", "M.Ga")
        .replace("DelanoeEtAl14", "D14")
        .replace("FieldEtAl07TR", "F07")
    )
    ax[0].set_prop_cycle(cycler("linestyle", linestyles))
    ax[0].plot(
        psd_size_grid,
        psd_data.T,
        c=colors[j],
        alpha=1,
        label=[
            f"{psd_label}, {t}K, {fwc_val*1e6:.1f}mg/m³"
            for t, fwc_val in zip(da_invonion_sel.Temperature.data, fwc)
        ],
    )
    ax[0].legend(loc="upper right", fontsize="small")

    psd_moment2 = psd_data * psd_size_grid**2
    psd_moment2_norm = psd_moment2.T / psd_moment2.sum(axis=1)
    ax[1].set_prop_cycle(cycler("linestyle", linestyles))
    ax[1].plot(
        psd_size_grid,
        psd_moment2_norm,
        c=colors[j],
        alpha=1,
        label=[
            f"{psd_label}, {t:.0f}K, {fwc_val*1e6:.1f}mg/m³"
            for t, fwc_val in zip(da_invonion_sel.Temperature.data, fwc)
        ],
    )
    ax[1].legend(loc="upper right", fontsize="small")
ax[0].set_ylabel("n(D)")
ax[0].set_title(f"dBZ={dBZ}dBZ")
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_ylim([1e6, 5e10])  # head
ax[0].grid()
ax[0].legend().set_visible(False)
ax[1].legend(title=da_invonion_sel.attrs["habit"])
ax[1].set_xlabel("Particle Size (m)")
ax[1].set_ylabel("D^2 * n(D) / integral")
ax[1].set_xscale("log")
ax[1].grid()

# %% 2nd moment integral vs dBZ
j = 0  # psd index
i = 0  # habit index
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 5))
for t_idx, t in enumerate([200, 220, 240]):
    for j in range(3):
        dBZ = [-20, -10, 0]
        da_invonion_sel = ds_onion_invtables[i, j].sel(
            Temperature=t
        )
        psd = da_invonion_sel.attrs["psd"]
        fwc = da_invonion_sel.pipe(lambda x: 10**x).data
        psd_size_grid, psd_data = get_psd(
            fwc=fwc,
            t=da_invonion_sel.Temperature.data * np.ones(len(fwc)),
            psd=psd,
        )
        moment2_integral = (psd_data * psd_size_grid**2).sum(axis=1)

        ax[t_idx].plot(
            da_invonion_sel.dBZ,
            moment2_integral,
            marker="o",
            label=psd,
        )
        ax[t_idx].grid()
        ax[t_idx].set_title(f"T={t}K")
        ax[t_idx].set_yscale("log")
        ax[t_idx].set_xlabel("dBZ")
        if t_idx == 0:
            ax[t_idx].set_ylabel("Integral of D^2 * n(D)")
        else:
            ax[t_idx].set_ylabel("")
ax[0].legend()
# %%
habit_idx = 0  # habit index
psd_idx = 0  # psd index
dBZ = [-20]
t = [240, 220, 200]
da_invonion_sel = ds_onion_invtables[habit_idx, psd_idx].sel(dBZ=dBZ, Temperature=t)
fwc = da_invonion_sel.pipe(lambda x: 10**x).data.squeeze()

if len(t) == 1 and len(fwc.squeeze()) != 1:
    t = t * len(fwc)

psd_size_grid, psd_data = get_psd(
    fwc=fwc,
    t=t,
    psd="DelanoeEtAl14",
    psd_size_grid=np.logspace(-6, -2, 100),
)
moment2 = psd_data * psd_size_grid**2
plt.plot(
    psd_size_grid,
    moment2.T / moment2.sum(axis=1),
    label=[
        f"{t:.0f}K, {fwc_val*1e6:.1f}mg/m³"
        for t, fwc_val in zip(da_invonion_sel.Temperature.data, fwc)
    ],
)


plt.xscale("log")
plt.legend(title=f"dBZ={dBZ[0]}dBZ")
plt.xlabel("Particle Size (m)")
plt.ylabel("D^2 * n(D) / integral")
plt.title(f"{da_invonion_sel.attrs['habit']}, {da_invonion_sel.attrs['psd']}")
plt.grid()

# %% plot an orbit
orbit_frame = "04549D"  # "04015F"#'04701A'#'03871C'#"04015F"

habits = []
psds = []
arts_T = []
arts_top_height = []
# Loop through all habit_std and psd combinations
i = 0  # habit index
for j in range(3):  # psd index
    habit_std, psd, orbits, ds_arts = load_arts_output_data(
        i, j, orbit_frame=orbit_frame, file_pattern=file_pattern
    )
    habits.append(habit_std)
    psds.append(psd)
    arts_T.append(ds_arts["arts"].mean("f_grid"))

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
# add cloud top height based on dbz threshold
for dbz_threshold in [-30, -25, -20]:
    ds_arts = get_cloud_top_height(
        ds_arts, dbz_threshold=dbz_threshold, based_on_fwc=False
    )
    ds_arts.cloud_top_height.plot(x="nray", ax=axes[0], label=f"dbz={dbz_threshold}")

    ds_arts = get_cloud_top_T(ds_arts, dbz_threshold=dbz_threshold, based_on_fwc=False)
    ds_arts.cloud_top_T.plot(x="nray", ax=axes[1], label=f"dbz={dbz_threshold}")
# axes[0].legend(loc="upper center")
# axes[1].legend(loc="upper center")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="center right",
    bbox_to_anchor=(1.05, 0.525),
    framealpha=0.3,
    # title=habit_std,
)
axes[1].grid()

colors = ["#0071B2FF", "#D55C00FF", "#009E74FF"]

# plot the difference between ARTS and MSI brightness temperature
kwargs = dict(ax=axes[2], x="nray", ls="-", lw=1, ylim=[-40, 40])
[
    (a - ds_arts["pixel_values"])
    .assign_attrs(units="K")
    .plot(label=f"{p}", c=c, **kwargs)
    for c, p, a in zip(colors, psds, arts_T)
]
(ds_arts["y_pred_ml"] - ds_arts["pixel_values"]).plot(label="XGBoost", c="r", **kwargs)
axes[2].axhline(0, color="k", ls="--", lw=1)

handles, labels = axes[2].get_legend_handles_labels()
labels_cust = ["D14", "F07", "M.Ga", "XGBoost", "MSI"]  # custom labels
fig.legend(
    handles,
    labels_cust,
    loc="center",
    bbox_to_anchor=(0.98, 0.225),
    framealpha=0.3,
    title=habit_std,
)
# optionally remove the axes legend:
axes[2].legend().set_visible(False)
axes[2].grid()

axes[0].set_title(f"Radar Reflectivity (CPR) - Orbit {orbit_frame}")
axes[1].set_title("Cloud Top Temperature")
axes[2].set_title("Predicted - True (MSI)")
axes[0].set_xlabel("")
axes[1].set_xlabel("")
axes[2].set_xlabel("Profile number")
axes[0].set_ylabel("z [km]")
axes[2].set_ylabel("Diff [K]")

# %%
