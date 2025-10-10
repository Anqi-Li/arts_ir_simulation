# %%
from pyarts.workspace import Workspace
import numpy as np
from earthcare_ir import PSD


# %%
def set_psd(ws, psd, mgd_coef=None):
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
    elif psd == PSD.F07M:
        ws.psdFieldEtAl07(
            scat_species_a=0.02,
            scat_species_b=2,
            regime="ML",
            # For low and high temperatures some parameters with default
            # values could matter
        )
    elif psd == PSD.MDG:
        if mgd_coef is not None:
            n0 = mgd_coef.get("n0")
            ga = mgd_coef.get("ga")
            mu = mgd_coef.get("mu")
        else:
            raise ValueError("For PSD.MDG, mgd_coef must be provided as a dict with keys 'n0', 'ga', 'mu'")
        ws.psdModifiedGammaMass(
            scat_species_a=0.02,
            scat_species_b=2,
            n0=n0,
            ga=ga,
            mu=mu,
            la=-999,
            t_max=373,
            t_min=0,
        )
    else:
        raise ValueError(f"PSD {psd} not handled")


def get_psd(fwc: np.array = None, t: np.array = None, psd=None, psd_size_grid=np.logspace(-6, -2, 100), mgd_coef=None):
    if len(fwc) != len(t):
        raise ValueError("fwc and t must have the same length")

    ws = Workspace(verbosity=0)
    ws.dpnd_data_dx_names = []
    ws.psd_size_grid = psd_size_grid  # Particle size grid in meters
    ws.pnd_agenda_input_t = t  # Temperature
    ws.pnd_agenda_input = fwc.reshape(-1, 1)  # FWC in kg/m3
    ws.pnd_agenda_input_names = ["FWC"]  # Name does not matter, just order!
    set_psd(ws, psd, mgd_coef=mgd_coef)

    return ws.psd_size_grid.value, ws.psd_data.value
