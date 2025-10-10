# %%
from earthcare_ir import  Habit, PSD

#%% investigate how many arts output files are there for different habits and psd 
def get_arts_output_files(orbit_frame, habit, psd):
    import data_paths as dp
    import os
    import glob
    path_pattern = os.path.join(
        dp.arts_output_TIR2,
        f'arts_TIR2_{orbit_frame}_{habit}_{psd}.nc'
    )
    return glob.glob(path_pattern)

arts_output_file_count = {
    (habit, psd): len(get_arts_output_files('*', habit, psd))
    for habit in [Habit.Bullet, Habit.Column, Habit.Plate]
    for psd in [PSD.MDG, PSD.F07T, PSD.D14]
}
arts_output_file_count
#%% check how many orbits are there in common orbit frame list
import data_paths as dp
orbit_list = dp.get_common_orbit_frame_list()
len(orbit_list)

# %% load onion invtables and show FWC vs Temperature for different dBZ
from cycler import cycler
from get_psd import get_psd
from onion_table import get_ds_onion_invtable
import numpy as np
import matplotlib.pyplot as plt

ds_onion_invtables = np.empty((3, 3), dtype=object)
fig, ax = plt.subplots(3, 3, figsize=(8, 8), sharey=True, sharex=True)
for i, habit in enumerate([Habit.Bullet, Habit.Column, Habit.Plate]):
    for j, psd in enumerate([PSD.MDG, PSD.F07T, PSD.D14]):
        print(f"Loading invtable for {habit}, {psd}...")

        if psd == PSD.MDG:
            # coefficients for the exponential PSD
            coef_mgd = {
                "n0": 1e10,  # Number concentration
                "ga": 1.5,  # Gamma parameter
            }
        else:
            coef_mgd = None
        ds_onion_invtable = get_ds_onion_invtable(
            habit=habit,
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

#%%
fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
# colors = ["b",  "r", "g","c", "m"]
colors = ["#0071B2FF", "#D55C00FF", "#009E74FF"]
linestyles = ["-", "--", ":", "-."]
for j in range(ds_onion_invtables.shape[1]):  # psd index
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
        mgd_coef=None if psd != PSD.MDG else {"n0": 1e10, "ga": 1.5, "mu": 0},
    )
    psd_label = (
        psd.replace(PSD.MDG, "M.Ga")
        .replace(PSD.D14, "D14")
        .replace(PSD.F07T, "F07")
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
    for j in range(ds_onion_invtables.shape[1]):  # psd index
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
            mgd_coef=None if psd != PSD.MDG else {"n0": 1e10, "ga": 1.5, "mu": 0},
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

