# %%
import os
from earthcare_ir import PSD, Habit, map_habit
import pyarts
import pyarts.xml as xml
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from get_psd import get_psd
from data_paths import (
    single_scattering_database_arts,
    single_scattering_database_PingYang,
)

# %% extract needed data
pingyang_database = False  # True: PingYang database, False: Arts standard database
habit = Habit.Column
if pingyang_database:
    datafolder = single_scattering_database_PingYang
    habit = map_habit[habit]  # habit name (string) in the database
else:
    datafolder = single_scattering_database_arts


def get_ss_db_in_arts_format(datafolder: str, habit: str):
    """
    datafolder: path (string) containing the single scattering database
    habit: habit name (string) in the database
    return: meta data and single scattering data loaded by pyarts.xml.load
    """
    M = xml.load(os.path.join(datafolder, f"{habit}.meta.xml"), search_arts_path=False)
    S = xml.load(os.path.join(datafolder, f"{habit}.xml"), search_arts_path=False)
    return M, S


def get_attr(obj, attr_name: str):
    """
    you can check what attr_names are available by:
    attributes = [attr for attr in dir(obj[0]) if not callable(getattr(obj[0], attr)) and not attr.startswith("__")]
    print("Attributes:", attributes)
    """
    return [getattr(obj[i], attr_name) for i in range(len(obj))]


# convert to xarray DataArray
def get_da(
    M,
    S,
    data_name: str,
    dims: list = ["f_grid", "T_grid", "za_inc_grid", "aa_inc_grid", "element"],
    attrs_dict: dict = {
        "long_name": None,
        "units": None,
        "habit": None,
        "source": None,
    },
    add_za_sca_grid: bool = False,
):
    assert len(M) == len(S)
    assert "za_sca_grid" in dims if add_za_sca_grid else True

    data = get_attr(S, data_name)
    f_grid = get_attr(S, "f_grid")
    longest_f_grid_idx = np.argmax([len(f_grid[i]) for i in range(len(f_grid))])  # pick the longest f_grid to be the common f_grid
    if add_za_sca_grid:
        za_grid = get_attr(S, "za_grid")
        longest_za_grid_idx = np.argmax(
            [len(za_grid[i]) for i in range(len(S))]
        )  # to add interpolate the za_grid when the length is different

    l_data_per_size = []
    for i in range(len(data)):
        data_per_size = (
            xr.DataArray(
                dims=dims,
                data=data[i],
                name=data_name,
                attrs={
                    "long_name": attrs_dict.get("long_name"),
                    "units": attrs_dict.get("units"),
                    "habit": attrs_dict.get("habit"),
                    "source": attrs_dict.get("source"),
                },
            )
            .assign_coords(
                {
                    "f_grid": ("f_grid", f_grid[i]),
                    "T_grid": ("T_grid", get_attr(S, "T_grid")[i]),
                }
            )
            .reindex(
                {"f_grid": f_grid[longest_f_grid_idx]},
                method="nearest",
                tolerance=0.01e9,  # 10 MHz tolerance for frequency alignment
                fill_value=np.nan,
            )
        )

        if add_za_sca_grid:
            data_per_size = data_per_size.assign_coords(
                {
                    "za_sca_grid": ("za_sca_grid", za_grid[i]),
                }
            ).interp(
                {"za_sca_grid": za_grid[longest_za_grid_idx]},
                method="nearest",  # interp to the longest za_grid
            )

        l_data_per_size.append(data_per_size)

    da = xr.concat(l_data_per_size, dim="size").assign_coords(
        {
            "d_veq": ("size", get_attr(M, "diameter_volume_equ")),
            "d_max": ("size", get_attr(M, "diameter_max")),
            "mass": ("size", get_attr(M, "mass")),
            "d_area": ("size", get_attr(M, "diameter_area_equ_aerodynamical")),
        }
    )

    da = da.assign_coords(
        {
            "wavelength": (
                "f_grid",
                pyarts.arts.convert.freq2wavelen(f_grid[longest_f_grid_idx]) * 1e6,
            )
        }
    )
    da["wavelength"] = da["wavelength"].assign_attrs({"long_name": "Wavelength", "units": "$\mu m$"})
    da["d_veq"] = da["d_veq"].assign_attrs({"long_name": "Volume equivalent sphere diameter", "units": "m"})
    da["d_max"] = da["d_max"].assign_attrs({"long_name": "Maximum dimension", "units": "m"})
    da["d_area"] = da["d_area"].assign_attrs({"long_name": "Aerodynamical area equivalent sphere diameter", "units": "m"})
    da["mass"] = da["mass"].assign_attrs({"long_name": "Particle mass", "units": "kg"})
    return da

def get_ab(d, m):
    # keep only positive finite values
    mask = np.isfinite(d) & np.isfinite(m) & (d > 0) & (m > 0)
    d1 = d[mask]; m1 = m[mask]
    logd = np.log(d1)
    logm = np.log(m1)
    b, loga = np.polyfit(logd, logm, 1)   # slope b, intercept loga
    a = np.exp(loga)
    return a, b

# %
# load single scattering database
M, S = get_ss_db_in_arts_format(datafolder, habit)

# convert to xarray DataArray
xs_ext = get_da(
    M,
    S,
    "ext_mat_data",
    dims=["f_grid", "T_grid", "za_inc_grid", "aa_inc_grid", "element"],
    attrs_dict={
        "long_name": "Extinction cross section",
        "units": "m2",
        "habit": habit,
        "source": datafolder,
    },
)

xs_abs = get_da(
    M,
    S,
    "abs_vec_data",
    dims=["f_grid", "T_grid", "za_inc_grid", "aa_inc_grid", "element"],
    attrs_dict={
        "long_name": "Absorption cross section",
        "units": "m2",
        "habit": habit,
        "source": datafolder,
    },
)

phase_matrix = get_da(
    M,
    S,
    "pha_mat_data",
    dims=[
        "f_grid",
        "T_grid",
        "za_sca_grid",
        "aa_sca_grid",
        "za_inc_grid",
        "aa_inc_grid",
        "element",
    ],
    attrs_dict={
        "long_name": "Phase matrix",
        "units": "m2",
        "habit": habit,
        "source": datafolder,
    },
    add_za_sca_grid=True,
)

a,b = get_ab(xs_ext.d_veq, xs_ext.mass)

# %%
# Plottings

# %% plot cross section
# xs = (xs_ext - xs_abs) / xs_ext
# xs = xs.assign_attrs(
#     {
#         "long_name": "Single scattering albedo",
#         "units": "1",
#         "habit": habit,
#         "source": datafolder,
#     }
# )
# xs = xs_ext / (xs_ext.d_veq) ** 2 / np.pi / 2
# xs = xs.assign_attrs(
#     {
#         "long_name": "Extinction cross section normalized by area",
#         "units": "m2/m2",
#         "expression": "$ \sigma_{ext}/d_{veq}^2$",
#         "habit": habit,
#         "source": datafolder,
#     }
# )

xs = (xs_ext / (xs_ext.mass)).isel(f_grid=slice(0, -1))
xs = xs.assign_attrs(
    {
        "long_name": "Extinction cross section normalized by mass",
        "units": "m2/kg",
        "expression": "$ \sigma_{ext}/m$",
        "habit": habit,
        "source": datafolder,
    }
)

# xs = xs_ext.assign_attrs(
#     {
#         "long_name": "Extinction cross section",
#         "units": "m2",
#         "expression": "$ \sigma_{ext}$",
#         "habit": habit,
#         "source": datafolder,
#     }
# )
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
skip_freq = 100 if pingyang_database else 5
xs.isel(
    f_grid=slice(None, None, skip_freq),
    T_grid=0,
    za_inc_grid=0,
    aa_inc_grid=0,
    element=0,
).plot(
    hue="f_grid",
    x="d_veq",
    xscale="log",
    yscale="log",
    add_legend=True,
    label=xs["f_grid"].data[::skip_freq] * 1e-9,
    ax=ax,
    ls="-",
    marker=".",
)

# xs.swap_dims({"f_grid": "wavelength"}).sel(wavelength=10.8, method="nearest").isel(T_grid=0, za_inc_grid=0, aa_inc_grid=0, element=0).plot(
#     x="d_veq",
#     xscale="log",
#     yscale="log",
#     ax=ax,
#     ls="--",
#     c="k",
#     label=10.8,
#     add_legend=True,
# )

handles, labels = ax.get_legend_handles_labels()
# labels = [f"{float(l):.1f} {xs['wavelength'].attrs['units']}" for l in labels]
labels = [f"{float(l):.1f} GHz" for l in labels]
ax.legend(handles, labels, title=f"{xs.attrs['habit']}")
ax.grid()
ax.set_title(f'{xs.attrs["long_name"]}')
ax.set_ylabel(f'{xs.attrs["expression"]} [{xs.attrs["units"]}]')
ax.figure.tight_layout()

plt.show()

# %% plot turnover
xs = (xs_ext / (xs_ext.mass)).drop_sel(f_grid=886.4e9).squeeze().isel(T_grid=2, size=slice(None, None, 5))
xs = xs.assign_attrs(
    {
        "long_name": "Extinction cross section normalized by mass",
        "units": "m2/kg",
        "expression": "$ \sigma_{ext}/m$",
        "habit": habit,
        "source": datafolder,
    }
)
xs["f_grid"] = xs["f_grid"].pipe(lambda x: x * 1e-9).assign_attrs({"units": "GHz", "long_name": "Frequency"})
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
xs.plot(
    x="f_grid",
    hue="d_veq",
    ls="-",
    marker=".",
    xscale="linear",
    yscale="linear",
    ax=ax,
    add_legend=True,
    label=xs["d_veq"].data,
)
handles, labels = ax.get_legend_handles_labels()
labels = [f"{float(l)*1e3:.1f} $mm$" for l in labels]
ax.legend(handles, labels, title=f"{xs.attrs['habit']}")
ax.grid()
ax.set_title(f'{xs.attrs["long_name"]} at {xs["T_grid"].data:.0f} K')
ax.set_ylabel(f'{xs.attrs["expression"]} [{xs.attrs["units"]}]')
ax.figure.tight_layout()

# %% plot backscattering phase matrix element
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
phase_matrix.isel(
    f_grid=slice(None, None, skip_freq),
    T_grid=0,
    aa_sca_grid=0,
    za_inc_grid=0,
    aa_inc_grid=0,
    element=0,
).sel(
    za_sca_grid=180,
    method="nearest",
).plot(
    hue="wavelength",
    x="d_veq",
    xscale="log",
    yscale="log",
    add_legend=True,
    label=phase_matrix["wavelength"].data[::skip_freq],
    ax=ax,
)
handles, labels = ax.get_legend_handles_labels()
labels = [f"{float(l):.1f} {phase_matrix['wavelength'].attrs['units']}" for l in labels]
ax.legend(handles, labels, title=f"{phase_matrix.attrs['habit']}")
ax.figure.tight_layout()
ax.grid()
ax.set_ylabel(f'{phase_matrix.attrs["long_name"]} [{phase_matrix.attrs["units"]}]')
plt.show()


# %% load psd
def get_psd_dataarray(psd_size_grid, fwc, t, psd, coef_mgd, scat_species_a=0.02, scat_species_b=2):
    psd_size_grid, psd_data = get_psd(
        fwc=fwc,
        t=t,
        psd=psd,
        psd_size_grid=psd_size_grid,
        mgd_coef=coef_mgd,
        scat_species_a=scat_species_a,
        scat_species_b=scat_species_b,
    )
    da_psd = xr.DataArray(
        data=psd_data,
        dims=["setting", "size"],
        coords={
            "d_veq": ("size", psd_size_grid),
            "fwc": ("setting", fwc),
            "temperature": ("setting", t),
        },
        attrs={
            "long_name": "Particle size distribution",
            "units": "m-4",
            "psd_type": psd,
            "coef_mgd": str(coef_mgd) if coef_mgd is not None else "None",
        },
    )  # .stack(setting=["fwc", "t"])
    da_psd["fwc"].attrs["long_name"] = "Ice water content"
    da_psd["fwc"].attrs["units"] = "kg/m3"
    da_psd["temperature"].attrs["long_name"] = "Temperature"
    da_psd["temperature"].attrs["units"] = "K"
    da_psd["d_veq"].attrs["long_name"] = "Volume equivalent sphere diameter"
    da_psd["d_veq"].attrs["units"] = "m"
    return da_psd


# %% plot phase matrix backscattering element weighted by psd
psd_size_grid = xs_ext["d_veq"].data
fwc = np.array([1e-3])  # IWC in kg/m3
t = np.array([220])  # Temperature in K
c = ["r", "g", "b"]
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
for i, psd in enumerate([PSD.D14, PSD.F07T, PSD.MDG]):

    coef_mgd = {"n0": 1e10, "ga": 1.5, "mu": 0} if psd == PSD.MDG else None
    da_psd = get_psd_dataarray(psd_size_grid, fwc, t, psd, coef_mgd, scat_species_a=a, scat_species_b=b)

    # % plot PSD moment6
    l_6 = (
        (da_psd * da_psd["d_veq"] ** 6)
        .pipe(lambda x: x / x.sum(dim="size"))
        .plot(
            x="d_veq",
            hue="setting",
            xscale="log",
            yscale="linear",
            ls=":",
            c=c[i],
            ax=ax,
            add_legend=False,
        )
    )

    # % plot phase matrix backscattering element weighted by PSD
    f_to_plot = 95 * 1e9  # GHz
    backscattering_efficiency = da_psd * (phase_matrix).isel(
        aa_sca_grid=0,
        za_inc_grid=0,
        aa_inc_grid=0,
        element=0,
    ).interp(
        za_sca_grid=180, method="nearest"
    ).sel(T_grid=t, method="nearest").sel(f_grid=f_to_plot, method="nearest")
    backscattering_efficiency.pipe(lambda x: x / x.sum(dim="size")).plot(
        x="d_veq",
        hue="setting",
        xscale="log",
        yscale="linear",
        label=psd,
        c=c[i],
        ls="-",
        add_legend=True,
        ax=ax,
    )

ax.set_ylabel("P11(180) * n(D) / norm")
ax.set_title(f"{backscattering_efficiency.f_grid.data*1e-9:.1f} GHz\n fwc={fwc.item():.0e} kg/m3\nT={t.item():.0f} K")
handles, labels = ax.get_legend_handles_labels()
handles.append(l_6[0])
labels.append("6th moment of n(D)")
ax.legend(handles, labels, title=f"{habit}")
ax.figure.tight_layout()
ax.grid(True)
plt.show()

#%% psd-weighted mass (bulk mass)
psd_size_grid = xs_ext["d_veq"].data
fwc = np.array([1e-3])  # IWC in kg/m3
t = np.array([220])  # Temperature in K
c = ["r", "g", "b"]
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
for i, psd in enumerate([PSD.D14, PSD.F07T, PSD.MDG]):
    coef_mgd = {"n0": 1e10, "ga": 1.5, "mu": 0} if psd == PSD.MDG else None
    da_psd = get_psd_dataarray(psd_size_grid, fwc, t, psd, coef_mgd, scat_species_a=a, scat_species_b=b)

    # 3rd moment of psd
    l_3 = (
        (da_psd * da_psd["d_veq"] ** 3)
        .pipe(lambda x: x / x.sum(dim="size"))
        .plot(
            x="d_veq",
            hue="setting",
            xscale="log",
            yscale="linear",
            ls=":",
            c=c[i],
            lw=5,
            ax=ax,
            add_legend=False,
        )
    )

    # % plot mass weighted by PSD
    mass_weighted = (da_psd * xs_ext.mass)
    mass_weighted.pipe(lambda x: x / x.sum(dim="size")).plot(
        x="d_veq",
        hue="setting",
        xscale="log",
        yscale="linear",
        label=psd,
        c=c[i],
        ls="-",
        add_legend=True,
        ax=ax,
    )
handles, labels = ax.get_legend_handles_labels()
ax.set_ylabel("m * n(D) / norm")
ax.set_title(f"Mass weighted by PSD\nfwc={fwc.item():.0e} kg/m3\nT={t.item():.0f} K")
ax.legend(handles, labels, title=f"{habit}")
ax.figure.tight_layout()
ax.grid(True)
plt.show()

# %% psd and plot psd-weighted cross section
psd_size_grid = xs_ext["d_veq"].data
fwc = np.array([10e-6])  # IWC in kg/m3
t = np.array([260])  # Temperature in K
c = ["r", "g", "b"]

fig, ax = plt.subplots(4, 1, figsize=(6, 3 * 4), sharex=True)
for i, psd in enumerate([PSD.D14, PSD.F07T, PSD.MDG]):
    # plot psd
    coef_mgd = {"n0": 1e10, "ga": 1.5, "mu": 0} if psd == PSD.MDG else None
    da_psd = get_psd_dataarray(psd_size_grid, fwc, t, psd, coef_mgd, scat_species_a=a, scat_species_b=b)
    da_psd.plot(
        x="d_veq",
        hue="setting",
        xscale="log",
        yscale="log",
        ax=ax[0],
        c=c[i],
        label=psd,
        add_legend=False,
    )

    # % plot PSD moment2
    # l_2 = (
    #     (da_psd * da_psd["d_veq"] ** 2)
    #     .pipe(lambda x: x / x.sum(dim="size"))
    #     .plot(
    #         x="d_veq",
    #         hue="setting",
    #         xscale="log",
    #         yscale="linear",
    #         ls=":",
    #         c=c[i],
    #         ax=ax[1],
    #         add_legend=False,
    #     )
    # )

    # % plot psd-weighted extinction cross section
    wavelength_to_plot = 10.8  # um
    psd_weighted_xs = da_psd * xs_ext.squeeze().sel(T_grid=t, method="nearest").swap_dims(
        {"f_grid": "wavelength"}
    ).sel(wavelength=wavelength_to_plot, method="nearest")
    l_ext = psd_weighted_xs.pipe(lambda x: x / x.sum(dim="size")).plot(
        x="d_veq",
        hue="setting",
        xscale="log",
        yscale="linear",
        label=psd,
        c=c[i],
        ls="-",
        add_legend=True,
        ax=ax[1],
    )

    psd_weighted_mass = da_psd * (xs_abs.mass)
    l_mass = psd_weighted_mass.pipe(lambda x: x / x.sum(dim="size")).plot(
        x="d_veq",
        hue="setting",
        xscale="log",
        yscale="linear",
        # label=psd,
        c=c[i],
        ls="-",
        add_legend=False,
        ax=ax[2],
    )

    xs_ext.squeeze().sel(T_grid=t, method="nearest").swap_dims({"f_grid": "wavelength"}).sel(
        wavelength=wavelength_to_plot, method="nearest"
    ).pipe(lambda x: x / x.mass).plot(
        x="d_veq",
        hue="setting",
        xscale="log",
        yscale="linear",
        c='k',
        ls="-",
        add_legend=False,
        ax=ax[3],
    )

ax[0].set_ylim([1e6, 1e11])
ax[0].set_ylabel("n(D)")
ax[0].grid(True)

ax[1].set_ylabel("$\sigma$ * n(D) / norm")
ax[1].set_title(f"psd-weighted $\sigma$ at {psd_weighted_xs.wavelength.data:.1f} um")
handles, labels = ax[1].get_legend_handles_labels()
ax[1].legend(handles, labels, title=f"{habit}\nT={t.item()-273:.0f}C, FWC={fwc.item()*1e6:.0f}mg/m3")
ax[1].figure.tight_layout()
ax[1].grid(True)

ax[2].set_ylabel("m * n(D) / norm")
ax[2].set_title(f"psd-weighted mass")
ax[2].grid(True)

ax[3].set_ylabel(f"$\sigma$/m")
ax[3].set_title(f"Extinction cross section normalized by mass")
ax[3].grid(True)
plt.show()

# %% plot bulk extinction with varied fwc and fixed t
psd_size_grid = xs_ext["d_veq"].data
c = ["r", "g", "b"]
fig, ax = plt.subplots(2, 1, figsize=(6, 4 * 2), sharex=False)
for i, psd in enumerate([PSD.D14, PSD.F07T, PSD.MDG]):
    coef_mgd = {"n0": 1e10, "ga": 1.5, "mu": 0} if psd == PSD.MDG else None
    fwc = np.logspace(0, 3) * 1e-6  # IWC in kg/m3
    t = 240  # Temperature in K
    da_psd = get_psd_dataarray(psd_size_grid, fwc, t * np.ones_like(fwc), psd, coef_mgd, scat_species_a=a, scat_species_b=b)

    psd_weighted_xs = (
        xs_ext.isel(za_inc_grid=0, aa_inc_grid=0, element=0)
        .sel(T_grid=t, method="nearest")
        .swap_dims({"f_grid": "wavelength"})
        .sel(wavelength=10.8, method="nearest")
        * da_psd
    )
    psd_weighted_xs.integrate(coord="d_veq").pipe(lambda x: x / x.fwc).plot(
        x="fwc",
        yscale="log",
        xscale="log",
        label=psd,
        c=c[i],
        ax=ax[0],
        add_legend=True,
    )

    psd_weighted_xs.d_veq[psd_weighted_xs.argmax("size")].pipe(lambda x: x * 1e6).plot(
        x="fwc", xscale="log", ax=ax[1], ls="-", c=c[i], add_legend=False
    )

ax[0].legend(title=f"T={t:.0f}K")
ax[0].set_title(
    f"{habit}, $\lambda$={xs_ext.wavelength.swap_dims({'f_grid':'wavelength'}).sel(wavelength=10.8,method='nearest').data:.1f} um"
)
ax[0].set_ylabel("$\int \sigma n(D) dD$/FWC [m2/kg]")
# ax[0].set_ylim([0, 200])
ax[0].grid(True, ls="--")
ax[1].set_title("D_veq at max contribution to extinction (um)")
ax[1].remove()
fig.tight_layout()
# %% plot psd from fixed dbz
from onion_table import get_ds_onion_invtable

psd_size_grid = xs_ext["d_veq"].data
fig, ax = plt.subplots(2, 1, figsize=(6, 4 * 2), sharex=True)
c = ["r", "g", "b", "m"]
for i, psd in enumerate([PSD.D14, PSD.F07T, PSD.MDG]):
    # psd = PSD.D14
    coef_mgd = {"n0": 1e10, "ga": 1.5, "mu": 0} if psd == PSD.MDG else None

    ds_onion_invtable = get_ds_onion_invtable(
        habit=Habit.Bullet,
        psd=psd,
        coef_mgd=coef_mgd,
    ).sel(radiative_properties="FWC", Temperature=[220], dBZ=[-20])

    # % plot psd-weighted cross section
    fwc = ds_onion_invtable.pipe(lambda x: 10**x).data.reshape(1)
    t = ds_onion_invtable.Temperature.data * np.ones_like(fwc)
    da_psd = get_psd_dataarray(psd_size_grid, fwc, t, psd, coef_mgd,scat_species_a=a, scat_species_b=b)
    psd_weighted_xs = (
        (
            da_psd
            * xs_ext.isel(
                T_grid=0,
                za_inc_grid=0,
                aa_inc_grid=0,
                element=0,
            )
            .swap_dims({"f_grid": "wavelength"})
            .sel(wavelength=10.8, method="nearest")  # select one wavelength,
        )
        .assign_attrs(
            {
                "long_name": "Extinction cross section weighted by PSD",
                "units": "m2 m-4",
                "coef_mgd": str(coef_mgd) if coef_mgd is not None else "None",
            }
        )
        .assign_coords(
            {
                "wavelength": 10.8,
                "habit": habit,
                "psd": psd,
            }
        )
    )
    # plot psd-weighted cross section
    psd_weighted_xs.pipe(lambda x: x / x.sum(dim="size")).plot(
        x="d_veq",
        hue="setting",
        xscale="log",
        yscale="linear",
        label=psd,
        add_legend=True,
        ax=ax[0],
        c=c[i],
    )

    # plot psd
    da_psd.plot(
        x="d_veq",
        hue="setting",
        xscale="log",
        yscale="log",
        ax=ax[1],
        c=c[i],
        label=psd,
        add_legend=False,
    )
ax[0].set_ylim([0, 0.04])
ax[0].set_ylabel("$\sigma_{{ext}}$ * n(D) / norm")
ax[0].set_xlabel("")
ax[0].set_title(ds_onion_invtable.attrs["habit"])
handles, labels = ax[0].get_legend_handles_labels()

ax[0].legend(
    handles,
    labels,
    title=f"Z={ds_onion_invtable.dBZ.data[0]} dBZ, T={ds_onion_invtable.Temperature.data[0]:.0f}K",
)
ax[0].grid(True)

ax[1].set_ylim([1e6, 1e11])
ax[1].set_ylabel("n(D)")
ax[1].set_title("Particle size distribution")
ax[1].grid(True)
plt.show()
# %% plot bulk extinction with fixed dbz
psd_size_grid = xs_ext["d_veq"].data
c = ["r", "g", "b"]
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
for i, psd in enumerate([PSD.D14, PSD.F07T, PSD.MDG]):
    coef_mgd = {"n0": 1e10, "ga": 1.5, "mu": 0} if psd == PSD.MDG else None
    ds_onion_invtable = (
        get_ds_onion_invtable(
            habit=Habit.Bullet,
            psd=psd,
            coef_mgd=coef_mgd,
        )
        .sel(radiative_properties="FWC", Temperature=210)
        .isel(dBZ=slice(None, None, 5))
    )
    fwc = ds_onion_invtable.pipe(lambda x: 10**x).data.squeeze()
    t = ds_onion_invtable.Temperature.data * np.ones_like(fwc)
    da_psd = get_psd_dataarray(psd_size_grid, fwc, t, psd, coef_mgd).assign_coords(
        {"dBZ": ("setting", ds_onion_invtable.dBZ.data.squeeze())}
    )

    (
        da_psd
        * xs_ext.isel(
            T_grid=0,
            za_inc_grid=0,
            aa_inc_grid=0,
            element=0,
        )
        .swap_dims({"f_grid": "wavelength"})
        .sel(wavelength=10.8, method="nearest")  # select one wavelength,
    ).integrate(coord="d_veq").plot(x="dBZ", yscale="log", label=psd, c=c[i], ax=ax)

plt.legend(title=f"{ds_onion_invtable.attrs['habit']}, T={ds_onion_invtable.Temperature.data:.0f}K")
plt.title("$\int \sigma_{ext}(\\lambda=10.8 \mu m, D) n(D) dD$")

# %% plot extinction cross section normalized by mass
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
xs_ext.isel(f_grid=slice(None, None, 5), T_grid=3).squeeze().pipe(lambda x: x / x.mass).plot(
    # x="wavelength",
    # x='f_grid',
    # hue="size",
    x="d_veq",
    hue="f_grid",
    xscale="log",
    yscale="log",
    add_legend=True,
    label=xs_ext["f_grid"].isel(f_grid=slice(None, None, 5)).data * 1e6,
    ax=ax,
)
# plt.legend(title="d_veq (um)")
handles, labels = ax.get_legend_handles_labels()
labels = [f"{float(l):.1f}" for l in labels]
ax.legend(handles, labels, title="d_veq ($\mu m$)")
ax.set_ylabel(f'{xs_ext.attrs["long_name"]}/mass [{xs_ext.attrs["units"]}/kg]')
ax.grid()
plt.show()
# %%
