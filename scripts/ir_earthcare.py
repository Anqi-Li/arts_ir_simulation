# %%
import pyarts
import numpy as np
import os
import xarray as xr
from datetime import datetime
from tqdm import tqdm
import easy_arts.wsv as wsv
import easy_arts.easy_arts as ea
from easy_arts.data_model import (
    CloudboxOption,
    SpectralRegion,
)
from physics.unitconv import specific_humidity2h2o_p
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

path_abs_lookup_table = os.path.join(
    os.getcwd(),
    "../data/lookup_tables/abs_table_Earthcare_TIR2_2025-06-12_16:37:12.618561.xml",
)

# % map the standard habit to the Yang habit name
map_habit = {
    "LargePlateAggregate": "Plate-Smooth",
    "8-ColumnAggregate": "8-ColumnAggregate-Smooth",
}


# %%
def cal_y_arts(ds_earthcare, habit_std, psd, ds_onion_invtable):
    # Create a workspace
    ws = ea.wsInit(SpectralRegion.TIR)

    # Download ARTS catalogs if they are not already present.
    pyarts.cat.download.retrieve()

    # Use a customised scattering folder to combine PingYang data and liquid droplet
    ws.stdhabits_folder = "/scratch/li/arts-yang-liquid-simlink"

    # The main agenda for computations of the radiative transfer
    # follows a pure clears sky radiative transfer in this example.
    # When the path tracing hits the surface, the surface emission
    # is computed from the lowest atmospheric temperature,
    # and when the path tracing hits the top of the atmosphere,
    # the cosmic background is used as emission source.
    ws.iy_main_agendaSet(option="Emission")
    ws.iy_surface_agendaSet(option="UseSurfaceRtprop")
    ws.surface_rtprop_agendaSet(option="Blackbody_SurfTFromt_field")
    ws.iy_space_agendaSet(option="CosmicBackground")

    # The path tracing is done step-by-step following a geometric path
    ws.ppath_agendaSet(option="FollowSensorLosPath")
    ws.ppath_step_agendaSet(option="GeometricPath")

    # We might have to compute the Jacobian for the retrieval
    # of relative humidity, so we need to set the agenda for
    # that.
    ws.water_p_eq_agendaSet(option="MK05")

    # The geometry of our planet, and several other properties, are
    # set to those of Earth.
    ws.PlanetSet(option="Earth")

    # Our output unit
    ws.iy_unit = "PlanckBT"  # "1"# W/(m^2 Hz sr)

    # We do not care about polarization in this example
    ws.stokes_dim = 1

    # The atmosphere is assumed 1-dimensional
    ws.atmosphere_dim = 1

    # We have one sensor position in space, looking down, and one
    # sensor position at the surface, looking up.
    ws.sensor_pos = [[300e3]]
    ws.sensor_los = [[180]]

    # The dimensions of the problem are defined by a 1-dimensional pressure grid
    ws.p_grid = ds_earthcare["pressure"].data  # Pa
    ws.lat_grid = []
    ws.lon_grid = []

    # Read absorption lookup table
    # interporlation orders (defaults are too high)
    ws.abs_nls_interp_order = 3
    ws.abs_t_interp_order = 3
    ws.abs_p_interp_order = 3
    ws.ReadXML(
        ws.abs_lookup,
        path_abs_lookup_table,
    )

    # Use the same absorption species and f_grid as in the lookup table
    ws.f_gridFromGasAbsLookup()
    # f_grid_abs_lookup = ws.abs_lookup.value.f_grid.value
    # ws.f_grid = f_grid_abs_lookup[::5]  # use every 5th frequency point
    ws.abs_species = ws.abs_lookup.value.species

    # Compute the absorption agenda.
    ws.abs_lines_per_speciesSetEmpty()
    ws.propmat_clearsky_agendaAuto(use_abs_lookup=1)
    ws.abs_lookupAdapt()

    # make atmosphere
    insert_atm_from_earthcare(ds_earthcare, ws)

    ws.jacobianOff()
    ws.sensorOff()

    # % Set the cloud layer
    scat_species = ["FWC", "LWC"]  # Arbitrary names
    insert_bulkprop_from_earthcare(ds_earthcare, ws, habit_std, psd, scat_species)

    # %
    # Cloudbox
    ea.scat_data(ws)
    ea.cloudbox(ws, CloudboxOption.AUTO)
    ws.pnd_fieldCalcFromParticleBulkProps()

    # We check the atmospheric geometry, the atmospheric fields, the cloud box,
    # and the sensor.
    ea.checks(ws)
    ea.disort(ws, Npfct=-1)

    # Switch off clouds?
    # ea.allsky2clearsky(ws)
    # print("All sky to clearsky done")

    # We perform the all sky calculations
    ws.yCalc()

    # save in data array
    return extract_xr_arrays(ds_earthcare, ws, scat_species)


def insert_bulkprop_from_earthcare(
    ds_earthcare, ws, habit_std, psd, scat_species=["FWC", "LWC"]
):
    for i, species in enumerate(scat_species):
        ws.Append(ws.particle_bulkprop_names, species)

        if species == "FWC":
            profile_fwc = ds_earthcare["frozen_water_content"].values
            wsv.particle_bulkprop_fieldInsert(ws, profile_fwc, i)

            ea.scat_data_rawAppendStdHabit(
                ws,
                habit=map_habit[habit_std],
                size_step=4,  # skip some particle sizes
            )

            # Set ice PSD
            if psd == "DelanoeEtAl14":
                ea.scat_speciesDelanoeEtAl14(ws, species)

            elif psd == "FieldEtAl07TR":
                ea.scat_speciesFieldEtAl07(ws, species, regime="TR")

            elif psd == "FieldEtAl07ML":
                ea.scat_speciesFieldEtAl07(ws, species, regime="ML")

            else:
                raise ValueError(f"PSD {psd} not handled")

        elif species == "LWC":
            # profile_lwc = ds_earthcare["cloud_liquid_water_content"].fillna(0).values
            profile_lwc = (
                ds_earthcare[["cloud_liquid_water_content", "rain_water_content"]]
                .fillna(0)
                .to_array()
                .sum("variable")
                .values
            )
            wsv.particle_bulkprop_fieldInsert(ws, profile_lwc, i)
            ea.scat_data_rawAppendStdHabit(
                ws,
                habit="MieSpheres_H2O_liquid",
                size_step=3,  # skip some particle sizes
                dmax_end=50e-3,  # ignore larger particles
            )
            ea.scat_speciesMgdMass(ws, species)

        else:
            raise ValueError(f"Scattering species {species} not handled")


def insert_atm_from_earthcare(ds_earthcare, ws):
    ws.z_field = ds_earthcare["height_grid"].data.reshape(-1, 1, 1)
    ws.t_field = ds_earthcare["temperature"].data.reshape(-1, 1, 1)
    if ds_earthcare["surfaceElevation"] > ds_earthcare["height_grid"].min():
        ws.z_surface = ds_earthcare["surfaceElevation"].data.reshape(1, 1)
    else:
        ws.z_surface = ds_earthcare["height_grid"].min().data.reshape(1, 1)

    vmr_h2o = (
        (
            specific_humidity2h2o_p(
                ds_earthcare["specific_humidity"],
                ds_earthcare["pressure"],
            )
            / ds_earthcare["pressure"]
        )
        .fillna(0)
        .data.reshape(-1, 1, 1)
    )

    vmr_co2 = 427.53e-6 * np.ones(len(ws.z_field.value)).reshape(-1, 1, 1)
    vmr_n2o = 330e-9 * np.ones(len(ws.z_field.value)).reshape(-1, 1, 1)
    vmr_o3 = (
        28.9644
        / 47.9982
        * ds_earthcare["ozone_mass_mixing_ratio"].data.reshape(-1, 1, 1)
    )
    vmr_field_stack = []
    for s in ws.abs_species.value:
        s = str(s).split("-")[0]
        if s == "H2O":
            vmr_field_stack.append(vmr_h2o)
        elif s == "CO2":
            vmr_field_stack.append(vmr_co2)
        elif s == "N2O":
            vmr_field_stack.append(vmr_n2o)
        elif s == "O3":
            vmr_field_stack.append(vmr_o3)
        else:
            raise (f"Absorption species {s} is not implemented")
    ws.vmr_field = np.stack(vmr_field_stack)
    return


def extract_xr_arrays(ds_earthcare, ws, scat_species):
    da_y = xr.DataArray(
        name="IR Temperature",
        data=ws.y.value,
        coords={
            "f_grid": pyarts.arts.convert.freq2wavelen(ws.f_grid.value.value) * 1e6
        },
        attrs={
            "unit": "K",
            "long_name": "Brightness Temperature",
        },
    ).sortby("f_grid")

    da_bulkprop_field = xr.DataArray(
        name="particle_bulkprob_field",
        # data=profile_fwc,
        data=ws.particle_bulkprop_field.value.value.squeeze(),
        coords={
            "scat_species": scat_species,
            "height_grid": ds_earthcare["height_grid"],
        },
        attrs={
            "unit": "kg m-3",
            # "long_name": "frozen water content",
        },
    )
    da_vmr_field = xr.DataArray(
        name="vmr_field",
        data=ws.vmr_field.value.value.squeeze(),
        coords={
            "abs_species": [str(s).split("-")[0] for s in list(ws.abs_species.value)],
            "height_grid": ds_earthcare["height_grid"],
        },
        attrs={
            "unit": "1",
            "long_name": "absoption species vmr",
        },
    )
    da_auxiliary = ds_earthcare[["time", "latitude", "longitude"]].copy()

    return da_y, da_bulkprop_field, da_vmr_field, da_auxiliary


# %%
def insert_fwc(ds_onion_invtable, ds_earthcare_):
    lowest_dBZ_threshold = -30
    ds_earthcare_["dBZ"] = ds_earthcare_["dBZ"].where(
        ds_earthcare_["dBZ"] >= lowest_dBZ_threshold
    )

    profile_fwc = (
        ds_onion_invtable.sel(radiative_properties="FWC")
        .interp(
            Temperature=ds_earthcare_["temperature"],
            dBZ=ds_earthcare_["dBZ"],
        )
        .pipe(lambda x: 10**x)  # Convert from log10(FWC) to FWC in kg/m^3
        .fillna(0)
    )

    ds_earthcare_ = ds_earthcare_.assign(
        frozen_water_content=(
            ds_earthcare_.dims,
            profile_fwc.where(
                profile_fwc >= 0,
                0,  # ensure no negative values
            ).data,
            dict(
                long_name="Frozen Water Content",
                units="kg m-3",
            ),
        )
    )

    return ds_earthcare_


if __name__ == "__main__":
    # %% Load Earthcare data

    # take a sample earthcare dataset
    path_earthcare = "../data/earthcare/arts_input_data/"
    orbit_frame = "01162E"
    ds_earthcare_ = xr.open_dataset(path_earthcare + f"arts_input_{orbit_frame}.nc")
    ds_earthcare_ = ds_earthcare_.rename({"z": "height_grid"})

    # % choose invtable
    habit_std = ["LargePlateAggregate", "8-ColumnAggregate"][1]  # Habit to use
    psd = ["DelanoeEtAl14", "FieldEtAl07TR", "FieldEtAl07ML"][0]  # PSD to use
    print(habit_std)
    print(psd)

    ds_onion_invtable = xr.open_dataset(
        os.path.join(
            os.getcwd(),
            f"../data/onion_invtables/onion_invtable_{habit_std}_{psd}.nc",
        ),
    )[f"onion_invtable_{habit_std}_{psd}"]

    # make FWC
    ds_earthcare_ = insert_fwc(ds_onion_invtable, ds_earthcare_)

    # remove clearsky profiles
    mask_low_fwc = (ds_earthcare_["frozen_water_content"] == 0).all(dim="height_grid")
    # mask_low_lwc = (ds_earthcare_["cloud_liquid_water_content"] == 0).all(
    #     dim="height_grid"
    # )
    # mask = np.logical_and(mask_low_fwc, mask_low_lwc)

    ds_earthcare_subset = ds_earthcare_.where(~mask_low_fwc, drop=True).isel(
        # nray=slice(None, None, 100)
    )
    print(f"Number of nrays: {len(ds_earthcare_subset.nray)}")

    # % loop over nrays
    def process_nray(i):
        return cal_y_arts(
            ds_earthcare_subset.isel(nray=i),
            habit_std,
            psd,
            ds_onion_invtable,
        )

    y = []
    bulkprop = []
    vmr = []
    auxiliary = []

    with ProcessPoolExecutor(
        max_workers=70
    ) as executor:  # Limit to 4 workers, adjust as needed
        futures = [
            executor.submit(process_nray, i)
            for i in range(len(ds_earthcare_subset.nray))
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc="process nrays"):
            da_y, da_bulkprop, da_vmr, da_auxiliary = f.result()
            y.append(da_y)
            bulkprop.append(da_bulkprop)
            vmr.append(da_vmr)
            auxiliary.append(da_auxiliary)

    da_y = xr.concat(y, dim="nray")
    da_bulkprop = xr.concat(bulkprop, dim="nray")
    da_vmr = xr.concat(vmr, dim="nray")
    da_auxiliary = xr.concat(auxiliary, dim="nray")

    ds_arts = (
        da_auxiliary.assign(
            arts=da_y,
            bulkprop=da_bulkprop,
            vmr=da_vmr,
        )
        .sortby("time")
        .assign(ds_earthcare_subset)
    )

    # % save to file
    ds_arts.to_netcdf(
        f"../data/earthcare/arts_output_data/{habit_std}_{psd}_{orbit_frame}_allsky.nc"
    )


# %% read a file
habit_std = "LargePlateAggregate"
psd = "DelanoeEtAl14"
orbit_frame = "01162E"
ds_arts = xr.open_dataset(
    f"../data/earthcare/arts_output_data/{habit_std}_{psd}_{orbit_frame}_allsky.nc"
)
# %%
# plot the results
fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 8), constrained_layout=True)
kwargs = dict(
    y="height_grid",
    x="nray",
    add_colorbar=True,
)
ds_arts["dBZ"].where(ds_arts["dBZ"] > -30).plot(ax=axes[0], vmin=-30, vmax=30, **kwargs)

ds_arts["bulkprop"].sel(scat_species="FWC").pipe(np.log10).plot(
    ax=axes[1], vmin=-6, vmax=-3, **kwargs
)
ds_arts["bulkprop"].sel(scat_species="LWC").pipe(np.log10).plot(ax=axes[2], **kwargs)

kwargs = dict(x="nray", marker=".", markersize=1, ax=axes[3])
ds_arts["arts"].mean("f_grid").plot(label="arts", **kwargs)
ds_arts["pixel_values"].plot(label="MSI", **kwargs)
axes[3].legend()

kwargs = dict(x="nray", marker=".", markersize=1, ax=axes[4])
ds_arts[["pixel_values", "arts"]].to_array().diff("variable").mean("f_grid").plot(
    **kwargs
)
axes[4].axhline(0, color="k", ls="--", lw=0.5)

axes[0].set_title("dBZ")
axes[1].set_title("FWC (log10)")
axes[2].set_title("LWC (log10)")
axes[3].set_title("Brightness Temperature (mean over f_grid)")
axes[4].set_title("ARTS - MSI")
axes[0].set_xlabel("")
axes[1].set_xlabel("")
axes[2].set_xlabel("")
axes[3].set_xlabel("")

fig.suptitle(
    f"""{habit_std}
{psd}
{orbit_frame}
    """,
)

# %%
# %% Plotting
# plt.figure()
# ds_arts.bulkprop.plot(y="height_grid", hue="nray", col="scat_species", sharex=False, xscale="log")
# plt.show()

# plt.figure()
# ds_arts.vmr.plot(y="height_grid", hue="nray", col="abs_species", sharex=False, xscale="log")
# plt.show()

# plt.figure()
# kwargs = dict(marker=".")
# ds_arts.arts.isel(f_grid=slice(None, None, 10)).mean("f_grid").plot(x="nray", label="arts 10th", **kwargs)
# ds_arts.arts.isel(f_grid=slice(None, None, 5)).mean("f_grid").plot(x="nray", label="arts 5th", **kwargs)
# ds_arts.arts.isel(f_grid=slice(None, None, 2)).mean("f_grid").plot(x="nray", label="arts 2th", **kwargs)
# ds_arts.arts.mean("f_grid").plot(x="nray", label="arts", **kwargs)

# ds_earthcare_subset["pixel_values"].plot(x="nray", label="MSI", **kwargs)
# # ds_earthcare_subset["temperature"].max(dim="height_grid").plot(x="nray", label="max T", **kwargs)
# ds_earthcare_subset.temperature.interp(
#     height_grid=ds_earthcare_subset.surfaceElevation,
#     kwargs={"fill_value": "extrapolate"},
# ).plot(x="nray", label="surface T (interpolate?)", **kwargs)
# plt.ylim([200, 302])
# plt.legend()
# plt.title(
#     f"""
# {habit_std}
# {psd}
# {orbit_frame}
#           """
# )
# plt.show()

# plt.figure()
# ds_arts.arts.plot(y="f_grid", x="nray")
