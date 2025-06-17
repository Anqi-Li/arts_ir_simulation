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
import matplotlib.pyplot as plt

# %% Load Earthcare data
# take a sample earthcare dataset
path_earthcare = "../data/earthcare/xy_data/"
orbit_frame = "01162E"
ds_earthcare_ = xr.open_dataset(path_earthcare + f"xy_{orbit_frame}.nc")
ds_earthcare_ = ds_earthcare_.eval("density = pressure/(temperature * 287e3)")
ds_earthcare_["density"] = ds_earthcare_["density"].assign_attrs(
    dict(
        long_name="Air Density",
        units="kg m-3",
    )
)
ds_earthcare_ = ds_earthcare_.eval(
    "cloud_liquid_water_content = specific_cloud_liquid_water_content * density"
)
ds_earthcare_["cloud_liquid_water_content"] = ds_earthcare_[
    "cloud_liquid_water_content"
].assign_attrs(
    dict(
        long_name="Liquid Water Content",
        units="kg m-3",
    )
)
# fill nan and inf
ds_earthcare_["dBZ"] = ds_earthcare_["dBZ"].fillna(-50)
ds_earthcare_["dBZ"] = ds_earthcare_["dBZ"].where(~np.isinf(ds_earthcare_["dBZ"]), -50)
mask_low_cloud = (ds_earthcare_.sel(height_grid=slice(10e3, None))["dBZ"] < -25).all(
    dim="height_grid"
)
ds_earthcare_ = ds_earthcare_.where(~mask_low_cloud, drop=True)

# %% choose invtable
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
    ws.p_grid = ds_earthcare["pressure"].__array__()  # Pa
    ws.lat_grid = []
    ws.lon_grid = []

    # Read lookup table
    # interporlation orders (defaults are too high)
    ws.abs_nls_interp_order = 3
    ws.abs_t_interp_order = 3
    ws.abs_p_interp_order = 3
    ws.ReadXML(
        ws.abs_lookup,
        os.path.join(
            os.getcwd(),
            "../data/lookup_tables/abs_table_Earthcare_TIR2_2025-06-12_16:37:12.618561.xml",
        ),
    )
    # The lookup table contains the absorption lines for the
    # gases in the atmosphere, and the frequencies at which
    # they are computed.  We can use this to set the frequency
    # grid for the absorption lines.
    # ws.f_gridFromGasAbsLookup()
    f_grid_abs_lookup = ws.abs_lookup.value.f_grid.value
    ws.f_grid = f_grid_abs_lookup[1::10]

    # Use the same absorption species as in the lookup table
    ws.abs_species = ws.abs_lookup.value.species

    # We now have all the information required to compute the absorption agenda.
    ws.abs_lines_per_speciesSetEmpty()
    ws.propmat_clearsky_agendaAuto(use_abs_lookup=1)
    ws.abs_lookupAdapt()

    # make atmosphere
    ws.z_field = ds_earthcare["height_grid"].__array__().reshape(-1, 1, 1)
    ws.t_field = ds_earthcare["temperature"].__array__().reshape(-1, 1, 1)
    if ds_earthcare["surfaceElevation"] > ds_earthcare["height_grid"].min():
        ws.z_surface = ds_earthcare["surfaceElevation"].__array__().reshape(1, 1)
    else:
        ws.z_surface = ds_earthcare["height_grid"].min().__array__().reshape(1, 1)

    vmr_h2o = (
        (
            specific_humidity2h2o_p(
                ds_earthcare["specific_humidity"],
                ds_earthcare["pressure"],
            )
            / ds_earthcare["pressure"]
        )
        .__array__()
        .reshape(-1, 1, 1)
    )

    vmr_co2 = 429e-6 * np.ones(len(ws.z_field.value)).reshape(-1, 1, 1)
    vmr_n2o = 330e-9 * np.ones(len(ws.z_field.value)).reshape(-1, 1, 1)
    ws.vmr_field = np.stack([vmr_h2o, vmr_co2, vmr_n2o])

    # These calculations do no partial derivatives, so we can turn it off
    ws.jacobianOff()

    # The concept of a sensor does not apply to this example, so we can turn it off
    ws.sensorOff()

    # % Set the cloud layer
    scat_species = ["FWC", "LWC"]  # Arbitrary names
    for i, species in enumerate(scat_species):

        ws.Append(ws.particle_bulkprop_names, species)

        if species == "FWC":
            # Create profile from earthcare radar by using the onion inversion table
            profile_earthcare_fwc_log10 = ds_onion_invtable.sel(
                radiative_properties="FWC"
            ).interp(
                dBZ=ds_earthcare["dBZ"],
                Temperature=ds_earthcare["temperature"],
                method="linear",
            )

            profile_earthcare_fwc = profile_earthcare_fwc_log10.pipe(
                lambda x: 10**x
            )  # Convert from log10(FWC) to FWC in kg/m^3
            profile_fwc = profile_earthcare_fwc.fillna(0).values
            wsv.particle_bulkprop_fieldInsert(ws, profile_fwc, i)

            # % map the standard habit name to the Yang habit name
            if habit_std == "LargePlateAggregate":
                habit = "Plate-Smooth"
            elif habit_std == "8-ColumnAggregate":
                habit = "8-ColumnAggregate-Smooth"

            ea.scat_data_rawAppendStdHabit(ws, habit=habit, size_step=4)

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
            profile_lwc = ds_earthcare["cloud_liquid_water_content"].fillna(0).values
            wsv.particle_bulkprop_fieldInsert(ws, profile_lwc, i)
            ea.scat_data_rawAppendStdHabit(ws, habit="MieSpheres_H2O_liquid")
            ea.scat_speciesMgdMass(ws, species)

        else:
            raise ValueError(f"Scattering species {species} not handled")

    # %
    # Cloudbox
    ea.scat_data(ws)
    ea.cloudbox(ws, CloudboxOption.AUTO)
    ws.pnd_fieldCalcFromParticleBulkProps()

    # We check the atmospheric geometry, the atmospheric fields, the cloud box,
    # and the sensor.
    ea.checks(ws)
    ea.disort(ws, Npfct=-1)
    # We perform the all sky calculations
    ws.yCalc()
    # save in data array
    da_y = xr.DataArray(
        name=habit_std,
        data=ws.y.value,
        coords={
            "f_grid": pyarts.arts.convert.freq2wavelen(ws.f_grid.value.value) * 1e6
        },
        attrs={
            "unit": "K",
        },
    ).sortby("f_grid")

    da_profile_fwc = xr.DataArray(
        name="fwc",
        data=profile_fwc,
        coords={"height_grid": ds_earthcare["height_grid"]},
        attrs={"unit": "kg m-3"},
    )

    return da_y, da_profile_fwc


# %%
y = []
fwc = []
for i in tqdm(range(len(ds_earthcare_.nray)), desc="process nrays"):
    # print(i)
    da_y, da_fwc = cal_y_arts(
        ds_earthcare_.isel(nray=i),
        habit_std,
        psd,
        ds_onion_invtable,
    )
    y.append(da_y)
    fwc.append(da_fwc)

da_y = xr.concat(y, dim="nray")
da_fwc = xr.concat(fwc, dim="nray")

ds = ds_earthcare_.assign(sim_y=da_y, fwc=da_fwc)

# %%
ds.to_netcdf(f"../data/earthcare/arts_y_data/y_lwc_{habit_std}_{psd}_{orbit_frame}.nc")


# %%
# da_y.mean("f_grid").plot(x="nray", label="arts")
# ds_earthcare_.isel(nray=slice(10))["pixel_values"].plot(x="nray", label="MSI")
# plt.legend()
#
# %%
ds_earthcare_.map_blocks(cal_y_arts, args=[habit_std, psd, ds_onion_invtable])


#  %%
def debug_cal_y_arts(
    ds_earthcare_,
    habit_std,
    psd,
    ds_onion_invtable,
    # surface,
):
    print(
        f"0: {ds_earthcare_.shape} | 1: {habit_std.shape} | 2: {psd.shape} | 3: {ds_onion_invtable.shape} "
    )
    return cal_y_arts(ds_earthcare_, habit_std, psd, ds_onion_invtable)


xr.apply_ufunc(
    debug_cal_y_arts,
    ds_earthcare_,
    habit_std,
    psd,
    ds_onion_invtable,
    # surface,
    input_core_dims=[["height_grid"], [], [], []],
    output_core_dims=[["f_grid"]],
    exclude_dims=set(("height_grid",)),
    vectorize=True,
    dask="parallelized",
    on_missing_core_dim="drop",
)
# %%
# ds_earthcare = ds_earthcare_.isel(nray=0)
# y, f_grid = cal_y_arts(ds_earthcare, habit_std, psd, ds_onion_invtable)

# da_y.f_grid.attrs["unit"] = "micrometer"


# from matplotlib import pyplot as plt

# da_y.plot()
# plt.hlines(y=ds_earthcare.pixel_values, xmin=10.35, xmax=11.25)

# # add shaded area between TIR bands
# plt.fill_betweenx(x1=8.35, x2=9.25, y=[215, 255], color="red", alpha=0.3, label="TIR 1")

# plt.fill_betweenx(x1=10.35, x2=11.25, y=[215, 255], color="green", alpha=0.3, label="TIR 2")

# plt.fill_betweenx(x1=11.55, x2=12.45, y=[215, 255], color="blue", alpha=0.3, label="TIR 3")
# plt.legend()

# %%
# orbit_frame = "01162E"
# habit_std = ["LargePlateAggregate", "8-ColumnAggregate"][0]  # Habit to use
# psd = ["DelanoeEtAl14", "FieldEtAl07TR", "FieldEtAl07ML"][0]  # PSD to use
# path_results = "../data/earthcare/arts_y_data/"
# filename = f"y_{habit_std}_{psd}_{orbit_frame}.nc"

# ds = xr.open_dataset(path_results + filename)

# fig, axes = plt.subplots(3, 1, sharex=True)


# ds["fwc"].where(ds["fwc"] > 0).pipe(np.log10).assign_attrs(long_name="Log10(FWC)").plot(
#     x="nray",
#     y="height_grid",
#     cbar_kwargs={"orientation": "horizontal", "aspect": 40},
#     vmin=-7,
#     vmax=-3,
#     ax=axes[0],
# )
# ds.rename(pixel_values="True", sim_y="ARTS")[["True", "ARTS"]].mean(
#     "f_grid"
# ).to_array().plot(
#     x="nray",
#     hue="variable",
#     ax=axes[1],
# )
# ds.rename(pixel_values="True", sim_y="ARTS")[["True", "ARTS"]].mean(
#     "f_grid"
# ).to_array().diff('variable').plot(
#     x="nray",
#     hue="variable",
#     ax=axes[2],
#     add_legend=False,
# )
# axes[1].set_ylabel('Tb [K]')
# axes[2].set_ylabel('Tb [K]')
# axes[2].set_title('ARTS - True')
# plt.suptitle(
#     f"""
# {habit_std}
# {psd}
# {orbit_frame}
# """
# )
# plt.tight_layout()
