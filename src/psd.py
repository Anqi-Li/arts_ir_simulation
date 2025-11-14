# %%
from pyarts.workspace import Workspace
import numpy as np
from earthcare_ir import PSD
import xarray as xr
from physics.constants import DENSITY_H2O_ICE, DENSITY_H2O_LIQUID


# %%
def set_ws_psd(
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


def get_ws_psd(
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
    set_ws_psd(
        ws,
        psd,
        mgd_coef=mgd_coef,
        scat_species_a=scat_species_a,
        scat_species_b=scat_species_b,
        rho=rho,
    )

    return ws.psd_size_grid.value, ws.psd_data.value


def get_psd_dataarray(
    psd_size_grid: np.ndarray,
    fwc: np.ndarray,
    t: np.ndarray,
    psd: str,
    coef_mgd: dict = None,
    scat_species_a: float = None,
    scat_species_b: float = None,
    convert_to_deq: bool = False,
    rho: float = DENSITY_H2O_LIQUID,
    varname_dgeo: str = "d_max",
    varname_deq: str = "d_meq",
) -> xr.DataArray:
    
    psd_size_grid, psd_data = get_ws_psd(
        fwc=fwc,
        t=t,
        psd=psd,
        psd_size_grid=psd_size_grid,
        mgd_coef=coef_mgd,
        scat_species_a=scat_species_a,
        scat_species_b=scat_species_b,
    )
    d_name = varname_deq if psd == PSD.D14 else varname_dgeo
    da_psd = xr.DataArray(
        data=psd_data,
        dims=["setting", "size"],
        coords={
            d_name: ("size", psd_size_grid),
            "fwc": ("setting", fwc),
            "temperature": ("setting", t),
        },
        attrs={
            "long_name": "Particle size distribution",
            "units": "m-4",
            "psd_type": psd,
            "coef_mgd": str(coef_mgd) if psd == PSD.MDG else "N/A",
            "a": scat_species_a if psd != PSD.D14 else "N/A",
            "b": scat_species_b if psd != PSD.D14 else "N/A",
        },
    )
    da_psd["fwc"] = da_psd["fwc"].assign_attrs(
        {"long_name": "Ice water content", "units": "kg/m3"}
    )
    da_psd["temperature"] = da_psd["temperature"].assign_attrs(
        {"long_name": "Temperature", "units": "K"}
    )
    da_psd[d_name] = (
        da_psd[d_name].assign_attrs(
            {
                "long_name": "Volume equivalent sphere diameter",
                "units": "m",
                "density": rho,
            }
        )
        if psd == PSD.D14
        else da_psd[d_name].assign_attrs(
            {
                "long_name": "Maximum dimension",
                "units": "m",
                "a": scat_species_a,
                "b": scat_species_b,
            }
        )
    )

    if convert_to_deq and psd != PSD.D14:
        da_psd = convert_psd_dgeo2deq(
            da_psd,
            a=scat_species_a,
            b=scat_species_b,
            rho=rho,
            varname_dgeo=varname_dgeo,
            varname_deq=varname_deq,
        )

    return da_psd


def convert_psd_dgeo2deq(
    da_psd: xr.DataArray,
    a: float,
    b: float,
    rho: float = DENSITY_H2O_LIQUID,
    varname_dgeo: str = "d_max",
    varname_deq: str = "d_meq",
) -> xr.DataArray:
    from physics.unitconv import dgeo2deq

    d_geo = da_psd[varname_dgeo].data
    d_eq, ddgeo_ddeq = dgeo2deq(dgeo=d_geo, a=a, b=b, rho=rho)
    da_psd_converted = da_psd * ddgeo_ddeq
    da_psd_converted = da_psd_converted.assign_attrs(da_psd.attrs)
    da_psd_converted = da_psd_converted.assign_coords(
        {varname_deq: ("size", d_eq)}
    ).drop_vars(varname_dgeo)
    da_psd_converted[varname_deq] = da_psd_converted[varname_deq].assign_attrs(
        {
            "long_name": "Volume equivalent sphere diameter",
            "units": "m",
            "density": rho,
        }
    )
    return da_psd_converted


def align_psd_size_grids(
    da_psd: xr.DataArray,
    d_name: str,
    common_size_grid: np.ndarray,
) -> xr.DataArray:
    return (
        da_psd.swap_dims({"size": d_name})
        .interp({d_name: common_size_grid}, kwargs={"fill_value": "extrapolate"})
        .swap_dims({d_name: "size"})
    )


# %% normalised PSD functions
def get_psd_moment_trapz(psd_data, size_grid, order, axis=-1, integrate=True):
    """Moment using trapezoidal rule.
    order: moment order
    psd_data: PSD data array
    size_grid: size grid array
    axis: axis along which to compute the moment
    """
    integrand = psd_data * size_grid**order
    if integrate:
        return np.trapezoid(integrand, size_grid, axis=axis)
    else:
        return integrand


def get_n0star(i, j, psd_data, size_grid):
    # Compute the normalized number concentration
    # N0/N0star = normalised F(x)
    n0star = get_psd_moment_trapz(psd_data, size_grid, i) ** (
        (j + 1) / (j - i)
    ) * get_psd_moment_trapz(psd_data, size_grid, j) ** ((i + 1) / (i - j))
    return n0star


def get_d0star(i, j, psd_data, size_grid):
    # Compute the normalized median diameter
    # x = d/d0star
    d0star = (
        get_psd_moment_trapz(psd_data, size_grid, i)
        / get_psd_moment_trapz(psd_data, size_grid, j)
    ) ** (-1 / (j - i))
    return d0star


# %% F05, ACM_CAP PSD functions
def psd_ACMCAP(n0star, d0star, dmeq):
    x = dmeq / d0star
    psd_data = n0star * psd_F05_norm(x)

    da_psd = xr.DataArray(
        data=psd_data,
        dims=["size"],
        coords={"d_meq": ("size", dmeq)},
        attrs={
            "long_name": "Particle size distribution ACM_CAP",
            "units": "m-4",
            "psd_type": "ACM_CAP",
        },
    )
    da_psd["d_meq"] = da_psd["d_meq"].assign_attrs(
        {"long_name": "Melted equivalent sphere diameter", "units": "m"}
    )
    return da_psd

def psd_F05_norm(x):
    # Following Field et al. (2005) Table 2 first row
    lambda_0 = 20.78
    lambda_1 = 3.29
    nu = 0.6357
    return 490.6 * np.exp(-lambda_0 * x) + 17.46 * (x**nu) * np.exp(-lambda_1 * x)


def psd_F05_prior(dmeq, t, iwc):
    n0star = get_n0star_prior_from_temperature(t=t)
    d0star = get_d0star_from_n0star_iwc(n0star=n0star, iwc=iwc)
    x = dmeq / d0star
    return n0star * psd_F05_norm(x)


def get_n0star_prior_from_temperature(t):
    # Following Mason et al 2023 Table 1
    t_c = t - 273.15
    a_v = np.exp(-6.9 + 0.0315 * t_c)
    n0prime = get_n0prime_prior_from_temperature(t)
    return n0prime * a_v**0.6


def get_n0prime_prior_from_temperature(t):
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


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example usage
    fwc = np.array([0.1])  # kg/m3
    t = np.array([220])  # K

    psd_size_grid = np.logspace(-6, -2, 100)  # m

    da_psd = get_psd_dataarray(
        psd_size_grid=psd_size_grid,
        fwc=fwc,
        t=t,
        psd=PSD.D14,
        coef_mgd=None,  # {"n0": 1e10, "ga": 1, "mu": 0},
        scat_species_a=None,
        scat_species_b=None,
        convert_to_dveq=True,
    )

    da_psd.plot(
        x="d_veq",
        hue="setting",
        yscale="linear",
        xscale="log",
        add_legend=False,
        figsize=(5, 4),
    )
    plt.ylabel("Number density per volume\nN(D) (m$^{-4}$)")
    plt.xlabel("D (m)")
    plt.grid()
    plt.gcf().patch.set_alpha(0)
    plt.show()

    # %%
    plt.figure(figsize=(5, 4))
    for n in range(0, 7):
        da_psd.pipe(
            get_psd_moment_trapz,
            order=n,
            size_grid=da_psd["d_veq"],
            integrate=False,
        ).pipe(lambda x: x / x.max()).plot(
            x="d_veq",
            hue="setting",
            yscale="linear",
            xscale="log",
            label=f"{n}th moment",
            add_legend=False,
        )
    plt.legend()
    plt.ylabel("the nth moment\nD$^{n} \\cdot N(D)$ (m$^{n-4}$)")
    plt.xlabel("D (m)")
    plt.grid()
    plt.gcf().patch.set_alpha(0)
    plt.show()

    # %%
    plt.text(
        0.5,
        0.5,
        r"$\propto \int D^n N(D) dD$",
        ha="center",
        va="center",
        fontsize=50,
    )
    plt.axis("off")
    plt.gcf().patch.set_alpha(0)
    plt.show()
