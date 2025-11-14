# %%
import os
from earthcare_ir import Habit, map_habit
import pyarts
import pyarts.xml as xml
import numpy as np
import xarray as xr
from data_paths import (
    single_scattering_database_arts,
    single_scattering_database_PingYang,
)


# %% extract needed data
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
    longest_f_grid_idx = np.argmax(
        [len(f_grid[i]) for i in range(len(f_grid))]
    )  # pick the longest f_grid to be the common f_grid
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
    da["wavelength"] = da["wavelength"].assign_attrs(
        {"long_name": "Wavelength", "units": "$\mu m$"}
    )
    da["d_veq"] = da["d_veq"].assign_attrs(
        {"long_name": "Volume equivalent sphere diameter", "units": "m"}
    )
    da["d_max"] = da["d_max"].assign_attrs(
        {"long_name": "Maximum dimension", "units": "m"}
    )
    da["d_area"] = da["d_area"].assign_attrs(
        {"long_name": "Aerodynamical area equivalent sphere diameter", "units": "m"}
    )
    da["mass"] = da["mass"].assign_attrs({"long_name": "Particle mass", "units": "kg"})
    return da


def get_mass_size_ab(d, m):
    # keep only positive finite values
    mask = np.isfinite(d) & np.isfinite(m) & (d > 0) & (m > 0)
    d1 = d[mask]
    m1 = m[mask]
    logd = np.log(d1)
    logm = np.log(m1)
    b, loga = np.polyfit(logd, logm, 1)  # slope b, intercept loga
    a = np.exp(loga)
    return a, b


def ss_data_extraction(datafolder: str, habit: str):
    M, S = get_ss_db_in_arts_format(datafolder, habit)
    xs_ext = get_da(
        M=M,
        S=S,
        data_name="ext_mat_data",
        dims=["f_grid", "T_grid", "za_inc_grid", "aa_inc_grid", "element"],
        attrs_dict={
            "long_name": "Extinction cross section",
            "units": "m2",
            "habit": habit,
            "source": datafolder,
        },
    )

    xs_abs = get_da(
        M=M,
        S=S,
        data_name="abs_vec_data",
        dims=["f_grid", "T_grid", "za_inc_grid", "aa_inc_grid", "element"],
        attrs_dict={
            "long_name": "Absorption cross section",
            "units": "m2",
            "habit": habit,
            "source": datafolder,
        },
    )

    phase_matrix = get_da(
        M=M,
        S=S,
        data_name="pha_mat_data",
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
    return xs_ext, xs_abs, phase_matrix


# %% test
if __name__ == "__main__":
    # example usage
    pingyang_database = False  # True: PingYang database, False: Arts standard database
    habit = Habit.Column
    if pingyang_database:
        datafolder = single_scattering_database_PingYang
        habit = map_habit[habit]  # habit name (string) in the database
    else:
        datafolder = single_scattering_database_arts

    xs_ext, xs_abs, phase_matrix = ss_data_extraction(datafolder, habit)
    a, b = get_mass_size_ab(xs_ext.d_veq, xs_ext.mass)

# %%
    for h in [Habit.Column, Habit.Bullet, Habit.Plate]:
        *_, phase_matrix = ss_data_extraction(datafolder, h)
        phase_matrix.sel(f_grid=95e9, za_sca_grid=180, method="nearest").isel(
            T_grid=0,
            za_inc_grid=0,
            aa_inc_grid=0,
            element=0,
        ).plot(
            x="d_veq",
            xscale="log",
            yscale="log",
            label=h,
            add_legend=True,
        )
