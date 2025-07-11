#!/usr/bin/env python

# Calculates a (non-complete) radar onion inversion table
# %%
import matplotlib.pyplot as plt
import numpy as np

import easy_arts.easy_arts as ea
from easy_arts.data_model import (
    SpectralRegion,
)
import xarray as xr
from select_habits import get_habit_list_from_path

# %%


def make_onion_invtable(habit, psd, coef_mgd=None):
    """
    Create a radar onion inversion table for the given habit and PSD.
    """
    # Start workspace
    ws = ea.wsInit(SpectralRegion.MICROWAVES)

    # Use the standard ARTS habit folder for MW regime
    ws.stdhabits_folder = "/scratch/patrick/Data/StandardHabits"
    # Check if the habit is available
    if habit not in get_habit_list_from_path(
        str(ws.stdhabits_folder), suffix=".xml.bin"
    ):
        raise ValueError(f"Habit {habit} not found in standard habits.")

    # We must define a RWC species, even though we will not use it
    ea.scat_speciesAbelBoutle12(ws, "RWC")
    ea.scat_data_rawAppendStdHabit(ws, habit="LiquidSphere")

    # Set an arbitrary frequency grid
    ws.f_grid = np.array([94e9])
    name = "FWC"  # Arbitrary species name

    # Set ice PSD
    if psd == "DelanoeEtAl14":
        ea.scat_speciesDelanoeEtAl14(ws, name)

    elif psd == "FieldEtAl07TR":
        ea.scat_speciesFieldEtAl07(ws, name, regime="TR")

    elif psd == "FieldEtAl07ML":
        ea.scat_speciesFieldEtAl07(ws, name, regime="ML")

    elif psd == "Exponential":
        # Define an exponential PSD with varing lambda
        if coef_mgd is None:
            raise ValueError("coef_mgd must be provided for Exponential PSD")
        n0 = coef_mgd.get("n0", 1e6)
        ga = coef_mgd.get("ga", 1)
        ea.scat_speciesMgdMass(ws, name, n0=n0, mu=0, la=-999, ga=ga)

    else:
        raise ValueError(f"PSD {psd} not handled")

    # Set ice habit
    ea.scat_data_rawAppendStdHabit(ws, habit=habit)

    # Form scat_data
    ws.scat_dataCalc()
    ws.scat_data_checkedCalc()

    # Calculate table
    #
    ws.ArrayOfGriddedField3Create("onion_invtable")
    ws.VectorCreate("onion_dbze_grid")
    ws.VectorCreate("onion_t_grid")
    #
    ws.VectorLinSpace(ws.onion_dbze_grid, -35, 20, 1)
    ws.VectorLinSpace(ws.onion_t_grid, 180, 273, 1)  # neglect temperatures above 0 C
    #
    ws.RadarOnionPeelingTableCalc(
        invtable=ws.onion_invtable,
        i_species=1,
        wc_min=1e-8,
        wc_max=2e-2,
        dbze_grid=ws.onion_dbze_grid,
        t_grid=ws.onion_t_grid,
    )

    return ws


# %% table in a data array and extrapolate to a wider dbz range
def get_ds_table(habit, psd, ws):
    ds_table = xr.DataArray(
        name=f"onion_invtable_{habit}_{psd}",
        data=ws.onion_invtable.value[1].data,
        coords={
            "radiative_properties": ["FWC", "Extinction"],
            "dBZ": ws.onion_dbze_grid.value,
            "Temperature": ws.onion_t_grid.value,
        },
        attrs={
            "unit": "log10(FWC/[kg/m3])",
            "description": "Radar onion inversion table for FWC",
        },
    )
    ds_table = ds_table.interp(
        dBZ=np.arange(-35, 35, 1),  # Extend the dBZ range
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )
    ds_table["Temperature"].attrs["unit"] = "K"
    ds_table["dBZ"].attrs["unit"] = "dBZ"
    return ds_table


# %%
def plot_table(ds_table):
    plt.figure()
    ds_table.sel(radiative_properties="FWC").isel(Temperature=[0, -1]).plot(
        x="dBZ",
        hue="Temperature",
        label="log10(FWC) [kg/m3]",
        ls="",
        marker=".",
    )
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # %% Create the inversion table
    for habit in ["LargePlateAggregate", "8-ColumnAggregate"]:  # Habit to use
        for psd in ["DelanoeEtAl14", "FieldEtAl07TR", "FieldEtAl07ML"]:  # PSD to use

            # Create the inversion table
            ws = make_onion_invtable(habit, psd)

            # % Create the data array with the inversion table
            ds_table = get_ds_table(habit, psd, ws)

            # plot_table(ds_table)

            # % Save the table to a netCDF file
            ds_table.to_netcdf(
                f"../data/onion_invtables/onion_invtable_{habit}_{psd}.nc",
                encoding={
                    "radiative_properties": {"dtype": "S1"},
                    "dBZ": {"dtype": "float32"},
                    "Temperature": {"dtype": "float32"},
                },
            )
    # %% take a sample earthcare dataset
    # path_earthcare = "../data/earthcare/arts_x_data/"
    # ds_earthcare = xr.open_dataset(path_earthcare + "arts_x_01162E.nc")

    # %% interpolate the table to the earthcare dataset
    # profile_fwc_log10 = ds_table.sel(radiative_properties="FWC").interp(
    #     Temperature=ds_earthcare.temperature,
    #     dBZ=ds_earthcare.dBZ,
    # )

    # fig, axes = plt.subplots(3,1, figsize=(12, 6), sharex=True, sharey=True)
    # profile_fwc_log10.plot(
    #     x="nray",
    #     y="height_grid",
    #     ax=axes[0],
    # )

    # ds_earthcare.dBZ.plot(
    #     x="nray",
    #     y="height_grid",
    #     ax=axes[1],
    # )

    # ds_earthcare.temperature.where((ds_earthcare.temperature<273)&(ds_earthcare.temperature>180)).plot(
    #     x="nray",
    #     y="height_grid",
    #     ax=axes[2],
    # )
