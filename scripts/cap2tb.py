#!/usr/bin/env python
# %%
# %load_ext autoreload
# %autoreload 2
import os
import numpy as np
import matplotlib.pyplot as plt

from ectools import ecio

import easy_arts.easy_arts as ea
import easy_arts.dataload as dl
import easy_arts.internal as internal
import easy_arts.wsv as wsv
import easy_arts.particle_models as pm
from easy_arts.data_model import (
    AbsSpeciesPredefinedOption,
    CloudboxOption,
    FascodVersion,
    IyUnit,
    ParticleModel,
    ParticleSizeDistribution,
    SkinTemperatureSource,
    SpectralRegion,
)

# %%
# Load a data frame and incorporate xmet data
#
import data_paths as dp

orbit_frame = "06203D"
dset = ecio.load_ACMCAP(
    srcpath=dp.ACMCAP,
    product_baseline="BA",
    orbit=orbit_frame[:-1],
    frame=orbit_frame[-1],
    nested_directory_structure=True,
)
dset.close()
#
ds_xmet = ecio.load_XMET(
    srcpath=dp.XMET,
    orbit=orbit_frame[:-1],
    frame=orbit_frame[-1],
)
ds_xmet.close()
dset = ecio.get_XMET(
    ds_xmet,
    dset,
    XMET_1D_variables=[],
    XMET_2D_variables=["temperature", "pressure", "specific_humidity"],
)

#%%
pmodels = [
    # LWC
    ParticleModel(
        psd=ParticleSizeDistribution.ACM_CAP_LWC,
        habit_folder="/scratch/li/arts-yang-liquid-simlink",
        habit_name="MieSpheres_H2O_liquid",
        habit_dmax_end=100e-6,
        habit_size_step=3,
    ),
    # FWC
    ParticleModel(
        psd=ParticleSizeDistribution.ACM_CAP_FWC,
        habit_folder="/scratch/li/arts-yang-liquid-simlink",
        # habit_name="8-ColumnAggregate-ModeratelyRough",
        habit_name="Plate-ModeratelyRough",
        # habit_name="SolidBulletRosette-ModeratelyRough",
        habit_size_step=4,
    ),
    # RWC
    ParticleModel(
        psd=ParticleSizeDistribution.ACM_CAP_RWC,
        habit_folder="/scratch/li/arts-yang-liquid-simlink",
        habit_name="MieSpheres_H2O_liquid",
        # habit_dmax_end=5e-3,
        habit_size_step=3,
    ),
]

# Switch to mono PSD for LWC?
if False:
    pmodels[0] = ParticleModel(
        psd=ParticleSizeDistribution.MONO_MASS_LWC,
        habit_folder="/scratch/li/arts-yang-liquid-simlink",
        habit_name="MieSpheres_H2O_liquid",
        habit_dmax_end=100e-6,
        habit_size_step=3,
    )

# %
# Start a ARTS calculations

# Basics
ws = ea.wsInit(SpectralRegion.TIR)
ws.SetNumberOfThreads(nthreads=8)
ea.atmosphereDim(ws, dim=1)
ea.planetEarth(ws)
ea.ppath(ws)
ea.rtEmission(ws, stokes=1)
#
ws.artsxmldata_folder = '/scratch/patrick/arts-xml-data-2.6/'
wsv.p_gridLog(ws, pmax=1050e2, pmin=50e2, nalt=201)

# Absorption
ws.abs_nls_interp_order = 3
ws.abs_t_interp_order = 3
ws.abs_p_interp_order = 3
path_abs_lookup_table = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data/lookup_tables/abs_table_Earthcare_TIR2_2025-06-12_16:37:12.618561.xml",
)
ws.ReadXML(
    ws.abs_lookup,
    path_abs_lookup_table,
)
ws.abs_species = ws.abs_lookup.value.species


# Sensor
ea.sensorNone(
    ws, f_grid=ws.abs_lookup.value.f_grid.value[::5], iy_unit=IyUnit.PLANCK_BT
)
wsv.sensor_poslos(ws, z=[300e3], za=[180])

# Compute the absorption agenda.
ws.abs_lines_per_speciesSetEmpty()
ws.propmat_clearsky_agendaAuto(use_abs_lookup=1)
ws.abs_lookupAdapt()

# Surface
ea.surface_typesInit(ws, single_type=True)
ea.surface_typesAppendBlackbody(ws, skin_t_source=SkinTemperatureSource.T_FIELD)

# Fixed parts of cloudbox
pm.to_scat_data(ws, pmodels)
ea.cloudbox(ws, CloudboxOption.FULL)


# Loop over profiles
pr0 = 2000 # First profile
prn = 3000  # Last profile
skip = 200
pr_fail = []
n = len(range(pr0, prn, skip))
print(f"Processing {n} profiles from {pr0} to {prn} with skip {skip}.")

y_as = np.full(n, np.nan)
for i, pr in enumerate(range(pr0, prn, skip)):
    ok = dl.ec_xmetAtmosphere1D(ws, dset, pr, FascodVersion.MLS)
    if ok:
        print(f"Processing profile {i}/{n}...")
        psd_input = dl.ec_acm_capHydrometeors1D(
            ws.p_grid.value,
            dset,
            pr,
        )

        pm.to_pnd_field_1d(ws, pmodels, psd_input)
        ea.checks(ws)
        ea.disort(ws, Npfct=-1)            
        ws.yCalc()
        y_as[i] = ws.y.value[0]



#% save for later comparison
y_as_save_fwc1 = y_as.copy()

#%%
ref = dset["MSI_longwave_brightness_temperature"].isel(
    along_track=slice(pr0, prn, skip), MSI_longwave_channel=1
).data
fig, ax = plt.subplots(1,1, figsize=(8,4))

ax.plot(np.arange(pr0, prn, skip), y_as_save_fwc - ref, label=f"ARTS 1 (Plate)", marker='x', alpha=0.7)
ax.plot(np.arange(pr0, prn, skip), y_as_save_fwc1 - ref, label=f"ARTS 1 (Plate)", marker='x', alpha=0.7)

# ax.plot(np.arange(pr0, prn, skip), y_as_save_fwc2 - ref, label=f"ARTS 2 (Plate)", marker='x', alpha=0.7)
# ax.plot(np.arange(pr0, prn, skip), y_as_save_fwc3 - ref, label=f"ARTS 3 (Plate)", marker='x', alpha=0.7)
# ax.plot(np.arange(pr0, prn, skip), y_as_save_fwc4 - ref, label=f"ARTS 4 (Plate)", marker='x', alpha=0.7)

dset = dset.eval(
    "Tb_diff = MSI_longwave_brightness_temperature_forward.isel(MSI_longwave_channel=1) - MSI_longwave_brightness_temperature.isel(MSI_longwave_channel=1)",
)
dset["Tb_diff"].isel(along_track=slice(pr0, prn, skip)).plot(
    ax=ax, label="ACMCAP forward", ls='--', c='grey',
)
ax.axhline(0, color='black', lw=0.5, ls='--')

ax.legend()
ax.set_xlabel("Along Track Index")
ax.set_ylabel("Tb difference (K)")
ax.set_title("ARTS - MSI brightness temperature differences")
plt.show()

# %%
fig, ax = plt.subplots(4, 1, figsize=(8, 6), sharex=True, constrained_layout=True)

# Rain water content
dset["rain_water_content"].isel(
    along_track=slice(pr0, prn, skip), JSG_height=slice(None, None)
).pipe(np.log10).plot(
    ax=ax[0],
    x="along_track",
    y="JSG_height",
    cmap="viridis",
    cbar_kwargs={"label": "log10 [kg/m3]"},
    add_colorbar=True,
)
ax[0].invert_yaxis()
ax[0].set_title('Rain Water Content')
ax[0].set_xlabel('')

# Liquid water content
dset["liquid_water_content"].isel(
    along_track=slice(pr0, prn, skip), JSG_height=slice(None, None)
).pipe(np.log10).plot(
    ax=ax[1],
    x="along_track",
    y="JSG_height",
    cmap="viridis",
    cbar_kwargs={"label": "log10 [kg/m3]"},
    add_colorbar=True,

)
ax[1].invert_yaxis()
ax[1].set_title('Liquid Water Content')
ax[1].set_xlabel('')

# Ice water content
dset["ice_water_content"].isel(
    along_track=slice(pr0, prn, skip), JSG_height=slice(None, None)
).pipe(np.log10).plot(
    ax=ax[2],
    x="along_track",
    y="JSG_height",
    cmap="viridis",
    cbar_kwargs={"label": "log10 [kg/m3]"},
)
ax[2].invert_yaxis()
ax[2].set_title('Ice Water Content')
ax[2].set_xlabel('')

# Brightness temperatures
dset["MSI_longwave_brightness_temperature"].isel(
    along_track=slice(pr0, prn, skip), MSI_longwave_channel=1
).plot(
    ax=ax[3],
    label="MSI",
    alpha=0.5,
)
dset["MSI_longwave_brightness_temperature_forward"].isel(
    along_track=slice(pr0, prn, skip), MSI_longwave_channel=1
).plot(
    ax=ax[3],
    label="ACMCAP forward",
    alpha=0.5,
)
ax[3].plot(np.arange(pr0, prn, skip), y_as, label=f"ARTS ({pmodels[1].habit_name})", marker='', alpha=0.7)

ax[3].legend()
ax[3].set_xlabel("Along Track Index")
ax[3].set_ylabel("Tb (K)")

fig.suptitle(f"ACM_CAP Profiles {orbit_frame}:BA")
plt.show()

# %%
# %%
# %%
print("ARTS simulated brightness temperatures:")
print(y_as)
print("MSI observed brightness temperatures:")
print(
    dset["MSI_longwave_brightness_temperature"]
    .isel(along_track=slice(pr0, prn, skip), MSI_longwave_channel=1)
    .data
)
print("ACMCAP forward model results:")
print(
    dset["MSI_longwave_brightness_temperature_forward"]
    .isel(along_track=slice(pr0, prn, skip), MSI_longwave_channel=1)
    .data
)

# Switch to clear-sky and rerun
# ea.allsky2clearsky(ws)
# for i, pr in enumerate(range(pr0,pr0+n)):
#     ok = dl.ec_xmetAtmosphere1D(ws, dset, pr, FascodVersion.MLS)
#     if ok:
#         ea.checks(ws)
#         ws.yCalc()
#         y_cs[i] = ws.y.value[0]

# print(y_cs-y_as)
# exit(1)
# plt.plot(y_cs)
# plt.plot(y_as)
# plt.show()