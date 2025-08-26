# %%
import numpy as np
import matplotlib.pyplot as plt
from pyarts.workspace import Workspace
import pyarts.xml as xml
import os

# %%
datafolder = "/data/s5/scattering_data/IRDbase/Yang2016/ArtsFormat"
habit = "8-ColumnAggregate-Smooth"
M = xml.load(os.path.join(datafolder, f"{habit}.meta.xml"), search_arts_path=False)
d_veq_Yang2016 = [getattr(M[i], "diameter_volume_equ", None) for i in range(len(M))]

# %% fixed psd, varying IWC and Temperature
ws = Workspace(verbosity=0)
ws.dpnd_data_dx_names = []
# ws.psd_size_grid = np.logspace(-5, -2, 21)
ws.psd_size_grid = np.array(d_veq_Yang2016)[::4]  # Use sizes from Yang2016
ws.pnd_agenda_input_t = np.array([260, 270])  # Temperature
ws.pnd_agenda_input = np.array([[1e-3], [1e-3]])  # IWC in kg/m3
ws.pnd_agenda_input_names = ["SWC"]  # Name does not matter, just order!


def set_psd(ws, psd):
    if psd == "f07":
        ws.psdFieldEtAl07(
            scat_species_a=0.02,
            scat_species_b=2,
            regime="TR",
            # For low and high temperatures some parameters with default
            # values could matter
        )
    elif psd == "mdg":
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
    elif psd == "delanoe":
        ws.psdDelanoeEtAl14(
            t_max=275,
            t_min=180,
            n0Star=-999,
            Dm=-999,
            alpha=-0.237,
            beta=1.839,
        )


psd = "delanoe"  # Options: 'f07', 'mdg', 'delanoe'
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

# %% fixed IWC and Temperature, varying psd
# ws.pnd_agenda_input_t = ws.pnd_agenda_input_t.value[:1]
# ws.pnd_agenda_input = ws.pnd_agenda_input.value[:1]
ws.pnd_agenda_input_t = np.array([230, 260])  # Temperature
ws.pnd_agenda_input = np.array([[1e-3], [1e-3]])  # IWC in kg/m3
line_styles = ["-", ":"]
colors = ["C0", "C1", "C2"]  # Use matplotlib's default color cycle
for i, psd in enumerate(["f07", "mdg", "delanoe"]):
    set_psd(ws, psd)
    # Plot each column (temperature) separately with different line styles but same color
    for j, temp in enumerate(ws.pnd_agenda_input_t.value):
        plt.plot(
            ws.psd_size_grid.value,
            ws.psd_data.value[j, :],
            ls=line_styles[j],
            color=colors[i],
            marker="o",
            label=f"{psd} T={temp}K, IWC={ws.pnd_agenda_input.value[j, 0]}kg/m3",
        )
plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.ylim([1e-20, 1e15])
plt.title("PSD for different model settings")
plt.xlabel("Diameter [m]")
plt.ylabel("PSD [m-1 m-3]")

# %%
