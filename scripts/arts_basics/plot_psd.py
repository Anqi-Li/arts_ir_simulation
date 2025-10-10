# %%
import numpy as np
import matplotlib.pyplot as plt
from pyarts.workspace import Workspace
import pyarts.xml as xml
import os
from earthcare_ir import PSD

save = False

# %%
datafolder = "/data/s5/scattering_data/IRDbase/Yang2016/ArtsFormat"
habit = "8-ColumnAggregate-Smooth"
M = xml.load(os.path.join(datafolder, f"{habit}.meta.xml"), search_arts_path=False)
d_veq_Yang2016 = [getattr(M[i], "diameter_volume_equ", None) for i in range(len(M))]


def set_psd(ws, psd):
    if psd == PSD.D14:
        ws.psdDelanoeEtAl14(
            t_max=275,
            t_min=180,
            n0Star=-999,
            Dm=-999,
            alpha=-0.237,
            beta=1.839,
        )
    elif psd == PSD.F07T:
        ws.psdFieldEtAl07(
            scat_species_a=0.02,
            scat_species_b=2,
            regime="TR",
            # For low and high temperatures some parameters with default
            # values could matter
        )
    elif psd == PSD.MDG:
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


# %% fixed IWC and Temperature, varying psd

ws = Workspace(verbosity=0)
ws.dpnd_data_dx_names = []
# ws.psd_size_grid = np.logspace(-5, -2, 21)
ws.psd_size_grid = np.array(d_veq_Yang2016)[::4]  # Use sizes from Yang2016
ws.pnd_agenda_input_t = np.array([210, 230, 260])  # Temperature
iwc_fixed = 1e-4  # Fixed IWC value in kg/m3
ws.pnd_agenda_input = iwc_fixed * np.ones([len(ws.pnd_agenda_input_t.value), 1])  # IWC in kg/m3
ws.pnd_agenda_input_names = ["FWC"]  # Name does not matter, just order!

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
line_styles = ["-", "--", ":"]
c = ["#0071B2FF", "#D55C00FF", "#009E74FF"]
for i, psd in enumerate([PSD.D14, PSD.F07T, PSD.MDG]):
    set_psd(ws, psd)
    # Plot each column (temperature) separately with different line styles but same color
    for j, temp in enumerate(ws.pnd_agenda_input_t.value):
        l = ax.plot(
            ws.psd_size_grid.value,
            ws.psd_data.value[j, :],
            ls=line_styles[j],
            color=c[i],
            alpha=0.8,
            label=f"{psd.replace('Exponential', 'MG').replace('DelanoeEtAl14', 'D14').replace('FieldEtAl07TR', 'F07TR')}, {temp:.0f}K",
        )
# ax.legend(loc="upper right", framealpha=0.3)
# handles, labels = ax.get_legend_handles_labels()
# labels[-3] = 'Mod. Gamma'
# fig.legend(handles[:-2], labels[:-2], loc="center right", bbox_to_anchor=(1.25, 0.6), framealpha=0.3)

ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylim([1e5, 1e11])
ax.set_xlim([1e-6, 1e-2])
ax.set_title(f"PSDs for FWC = 10^{int(np.log10(iwc_fixed))} kg/m3")
ax.set_xlabel("Diameter Volume-eq. [m]")
ax.set_ylabel("PSD [m-1 m-3]")

if save:
    fig.savefig(f"../data/figures/psd_comparison_fixedIWC_{int(np.log10(iwc_fixed))}.png", dpi=400, bbox_inches="tight")
    save = False
# %% fixed psd, varying IWC and Temperature
ws = Workspace(verbosity=0)
ws.dpnd_data_dx_names = []
# ws.psd_size_grid = np.logspace(-5, -2, 21)
ws.psd_size_grid = np.array(d_veq_Yang2016)[::4]  # Use sizes from Yang2016
ws.pnd_agenda_input_t = np.array([260, 270])  # Temperature
ws.pnd_agenda_input = np.array([[1e-3], [1e-3]])  # IWC in kg/m3
ws.pnd_agenda_input_names = ["FWC"]  # Name does not matter, just order!

psd = PSD.D14
set_psd(ws, psd)

plt.figure()
plt.plot(
    ws.psd_size_grid.value,
    ws.psd_data.value.T,
    marker="o",
    label=[
        f"{t}[K], IWC = {iwc}[kg/m3]"
        for t, iwc in zip(
            ws.pnd_agenda_input_t.value,
            ws.pnd_agenda_input.value,
        )
    ],
)
plt.title(f"PSD {psd}")
plt.xscale("linear")
plt.yscale("log")
plt.legend()
plt.xlabel("Diameter [m]")
plt.ylabel("PSD [m-1 m-3]")
