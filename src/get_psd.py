# %%
from pyarts.workspace import Workspace
import numpy as np
from earthcare_ir import PSD


# %%
def set_psd(
    ws,
    psd,
    mgd_coef=None,
    scat_species_a=0.02,
    scat_species_b=2,
    rho=1000,
):
    if psd == PSD.D14:
        ws.psdDelanoeEtAl14(
            t_max=275,
            t_min=180,
            n0Star=-999,  # calculated from temperature internally
            Dm=-999,
            alpha=-0.237,  # can be changed to other DARDAR version values if needed
            beta=1.839,
            rho=rho,
        )
    elif psd == PSD.F07T:
        ws.psdFieldEtAl07(
            scat_species_a=scat_species_a,
            scat_species_b=scat_species_b,
            regime="TR",
        )
    elif psd == PSD.F07M:
        ws.psdFieldEtAl07(
            scat_species_a=scat_species_a,
            scat_species_b=scat_species_b,
            regime="ML",
        )
    elif psd == PSD.MDG:
        if mgd_coef is not None:
            n0 = mgd_coef.get("n0")
            ga = mgd_coef.get("ga")
            mu = mgd_coef.get("mu")
        else:
            raise ValueError(
                "For PSD.MDG, mgd_coef must be provided as a dict with keys 'n0', 'ga', 'mu'"
            )
        ws.psdModifiedGammaMass(
            scat_species_a=scat_species_a,
            scat_species_b=scat_species_b,
            n0=n0,
            ga=ga,
            mu=mu,
            la=-999,
            t_max=373,
            t_min=0,
        )
    else:
        raise ValueError(f"PSD {psd} not handled")


def get_psd(
    fwc: np.array = None,
    t: np.array = None,
    psd=None,
    psd_size_grid=np.logspace(-6, -2, 100),
    mgd_coef=None,
    scat_species_a=0.02,
    scat_species_b=2,
    rho=1000,
):
    if isinstance(fwc, np.ndarray) and isinstance(t, np.ndarray) and len(fwc) != len(t):
        raise ValueError("fwc and t must have the same length")
    elif isinstance(fwc, float) and isinstance(t, float):
        fwc = np.array([fwc])
        t = np.array([t])
    ws = Workspace(verbosity=0)
    ws.dpnd_data_dx_names = []
    ws.psd_size_grid = psd_size_grid  # Particle size grid in meters
    ws.pnd_agenda_input_t = t  # Temperature
    ws.pnd_agenda_input = fwc.reshape(-1, 1)  # FWC in kg/m3
    ws.pnd_agenda_input_names = ["FWC"]  # Name does not matter, just order!
    set_psd(
        ws,
        psd,
        mgd_coef=mgd_coef,
        scat_species_a=scat_species_a,
        scat_species_b=scat_species_b,
        rho=rho,
    )

    return ws.psd_size_grid.value, ws.psd_data.value

# %% normalised PSD functions
def get_psd_moment_trapz(order, psd_data, size_grid, axis=-1):
    """Moment using trapezoidal rule.
    order: moment order
    psd_data: PSD data array
    size_grid: size grid array
    axis: axis along which to compute the moment
    """
    integrand = psd_data * size_grid**order
    return np.trapezoid(integrand, size_grid, axis=axis)


def get_n0star(i, j, psd_data, size_grid):
    # Compute the normalized number concentration
    # N0/N0star = normalised F(x)
    n0star = get_psd_moment_trapz(i, psd_data, size_grid) ** (
        (j + 1) / (j - i)
    ) * get_psd_moment_trapz(j, psd_data, size_grid) ** ((i + 1) / (i - j))
    return n0star


def get_d0star(i, j, psd_data, size_grid):
    # Compute the normalized median diameter
    # x = d/d0star
    d0star = (
        get_psd_moment_trapz(i, psd_data, size_grid)
        / get_psd_moment_trapz(j, psd_data, size_grid)
    ) ** (-1 / (j - i))
    return d0star


#%% F05, EarthCARE PSD functions
def psd_F05_norm(x):
    # Following Field et al. (2005) Table 2 first row
    lambda_0 = 20.78
    lambda_1 = 3.29
    nu = 0.6357
    return (490.6 * np.exp(-lambda_0 * x) + 17.46 * (x**nu) * np.exp(-lambda_1 * x))
    
def psd_F05(dmeq, t, iwc):
    n0star = get_n0star_from_temperature(t=t)
    d0star = get_d0star_from_n0star_iwc(n0star=n0star, iwc=iwc)
    x = dmeq / d0star
    return n0star * psd_F05_norm(x)

def get_n0star_from_temperature(t):
    # Following Mason et al 2023 Table 1
    t_c = t - 273.15
    a_v = np.exp(-6.9 + 0.0315 * t_c)
    n0prime = get_n0prime_from_temperature(t)
    return n0prime * a_v**0.6

def get_n0prime_from_temperature(t):
    t_c = t - 273.15
    return np.exp(16.118 - 0.1303 * t_c)

def get_d0star_from_n0star_iwc(iwc, n0star):
    # Compute d0star from n0star and iwc
    # Following Delanoë et al. (2014) Eq. (9)
    # Using the relation: iwc = (pi/6) * rho * n0star * d0star^4 * C
    # where C is a constant, arbitrarily chosen by Testud et al. (2001) to be equal to gamma(4)/4^4

    rho = 1000  # kg/m^3  # used in Delanoë et al. (2014)
    # C = math.gamma(4) / 4**4  # constant from Testud et al. (2001)
    C = 1
    d0star = (6 * iwc / (np.pi * rho * n0star * C)) ** (1 / 4)
    return d0star
