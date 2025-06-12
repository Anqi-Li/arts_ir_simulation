# %%
import pyarts
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
import time

# %%
# Create a workspace
ws = pyarts.Workspace()

# Download ARTS catalogs if they are not already present.
pyarts.cat.download.retrieve()

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
ws.p_grid = np.logspace(np.log10(1050e2), np.log10(10e2), 150)  # Pa
ws.lat_grid = []
ws.lon_grid = []

# The surface is at 0-meters altitude
ws.z_surface = [[0.0]]

# Read lookup table
# interporlation orders (defaults are too high)
ws.abs_nls_interp_order = 3
ws.abs_t_interp_order = 3
ws.abs_p_interp_order = 3
ws.ReadXML(
    ws.abs_lookup,
    os.path.join(
        os.getcwd(),
        "data/lookup_tables/lookup_table_perturb14_2025-03-13_11:29:32.528647.xml",
    ),
)

ws.f_gridFromGasAbsLookup()

# The atmosphere consists of water, oxygen and nitrogen.
# We set these to be computed using predefined absorption
# models.
ws.abs_speciesSet(
    species=[
        "H2O, H2O-SelfContCKDMT400 , H2O- ForeignContCKDMT400",
        "O3",
        "CO2, CO2-CKDMT252",
        "CH4",
        "N2O",
    ]
)

# We now have all the information required to compute the absorption agenda.
ws.abs_lines_per_speciesSetEmpty()
ws.propmat_clearsky_agendaAuto(use_abs_lookup=1)
ws.abs_lookupAdapt()

# We need an atmosphere.  This is taken from the arts-xml-data in
# raw format before being turned into the atmospheric fields that
# ARTS uses internally.
ws.AtmRawRead(basename="planets/Earth/Fascod/midlatitude-winter/midlatitude-winter")
ws.AtmFieldsCalc()

# These calculations do no partial derivatives, so we can turn it off
ws.jacobianOff()

# There is no scattering in this example, so we can turn it off
ws.cloudboxOff()

# The concept of a sensor does not apply to this example, so we can turn it off
ws.sensorOff()

# We check the atmospheric geometry, the atmospheric fields, the cloud box,
# and the sensor.
ws.atmgeom_checkedCalc()
ws.atmfields_checkedCalc()
ws.cloudbox_checkedCalc()
ws.sensor_checkedCalc()

# We perform the calculations
start = time.time()
ws.yCalc()
end = time.time()
time_delta = end - start
print(time_delta, "s")

# order=3, perturb=4: 0.029036998748779297 s
# order=3, perturb=14: 0.015968799591064453 s
# order=7, perturb=14: 0.07083249092102051 s

# %% saving dataset
simulation_tag = "perturb14_order3_noO3"
xr.Dataset(
    data_vars={simulation_tag: ("f_grid", ws.y.value)},
    coords={"f_grid": pyarts.arts.convert.freq2kaycm(ws.f_grid.value.value)},
).to_netcdf(
    os.path.join(os.getcwd(), "data/y_{}.nc".format(simulation_tag)),
)
