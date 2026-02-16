#!/usr/bin/env python
"""
ARTS Simulation for AWS and IR Brightness Temperatures
Compute ARTS simulated brightness temperatures for given EarthCARE orbit.

Can be used interactively or as a CLI script:
    python cap2tb_aws_ir.py --orbit 06203 --frame D --plot
    python cap2tb_aws_ir.py --orbit 06203 --frame D --no-plot --output results.nc
"""
# %%
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pyarts
from tqdm import tqdm
import xarray as xr

from ectools import ecio

import easy_arts.easy_arts as ea
import easy_arts.dataload as dl
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
import data_paths as dp
import sensor.aws
import sensor.hmatrix as hmatrix

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================


def load_data(orbit, frame):
    """Load ACMCAP, XMET, and CFMR datasets for given orbit/frame."""
    print(f"Loading data for orbit {orbit}, frame {frame}...")

    dset = ecio.load_ACMCAP(
        srcpath=dp.ACMCAP,
        product_baseline="BA",
        orbit=orbit,
        frame=frame,
        nested_directory_structure=True,
    )
    dset.close()

    ds_xmet = ecio.load_XMET(
        srcpath=dp.XMET,
        product_baseline="AA",
        orbit=orbit,
        frame=frame,
    )
    ds_xmet.close()

    dset = ecio.get_XMET(
        ds_xmet,
        dset,
        XMET_1D_variables=[],
        XMET_2D_variables=[
            "temperature",
            "pressure",
            "specific_humidity",
            "ozone_mass_mixing_ratio",
        ],
    )

    dset_fmr = ecio.load_CFMR(
        srcpath=dp.CFMR,
        product_baseline="BA",
        orbit=orbit,
        frame=frame,
        nested_directory_structure=True,
    )
    dset_fmr.close()
    dset = dset.update(dset_fmr[["land_flag"]])

    return dset, ds_xmet, dset_fmr


def init_workspace():
    """Initialize ARTS workspace with basic settings."""
    print("Initializing ARTS workspace...")
    ws = ea.wsInit(SpectralRegion.IGNORE)
    ws.SetNumberOfThreads(nthreads=8)
    ea.atmosphereDim(ws, dim=1)
    ea.planetEarth(ws)
    ea.ppath(ws)
    ea.rtEmission(ws, stokes=1)
    ws.artsxmldata_folder = "/scratch/patrick/arts-xml-data-2.6/"
    wsv.p_gridLog(ws, pmax=1050e2, pmin=50e2, nalt=201)

    # Surface
    ea.surface_typesInit(ws, single_type=True)
    ea.surface_typesAppendBlackbody(ws, skin_t_source=SkinTemperatureSource.T_FIELD)

    return ws


def get_default_pmodels_ir(
    scat_data_folder="/scratch/li/arts-yang-liquid-simlink",
    ice_habit="8-ColumnAggregate-ModeratelyRough",
):
    """Return default particle models for IR simulation."""
    return [
        # LWC
        ParticleModel(
            psd=ParticleSizeDistribution.ACM_CAP_LWC,
            habit_folder=scat_data_folder,
            habit_name="MieSpheres_H2O_liquid",
            habit_dmax_end=1e-3,  # check definition of liquid in captivate?
            # habit_size_step=3,
        ),
        # FWC
        ParticleModel(
            psd=ParticleSizeDistribution.ACM_CAP_FWC,
            habit_folder=scat_data_folder,
            habit_name=ice_habit,
            # habit_size_step=3,
        ),
        # RWC
        ParticleModel(
            psd=ParticleSizeDistribution.ACM_CAP_RWC,
            habit_folder=scat_data_folder,
            habit_name="MieSpheres_H2O_liquid",
            habit_dmax_end=7e-3,  # physical upper limit for rain
            # habit_size_step=3,
        ),
    ]


def get_default_pmodels_aws(
    scat_data_folder=dp.single_scattering_database_arts,
    ice_habit="LargePlateAggregate",
):
    """Return default particle models for AWS simulation."""
    return [
        # LWC
        ParticleModel(
            psd=ParticleSizeDistribution.ACM_CAP_LWC,
            habit_folder=scat_data_folder,
            habit_name="LiquidSphere",
            habit_dmax_end=5e-4,  # check definition of liquid in captivate?
            # habit_size_step=3,
        ),
        # FWC
        ParticleModel(
            psd=ParticleSizeDistribution.ACM_CAP_FWC,
            habit_folder=scat_data_folder,
            habit_name=ice_habit,
            # habit_size_step=3,
        ),
        # RWC
        ParticleModel(
            psd=ParticleSizeDistribution.ACM_CAP_RWC,
            habit_folder=scat_data_folder,
            habit_name="LiquidSphere",
            habit_dmax_end=7e-3,  # physical upper limit for rain
            # habit_size_step=3,
        ),
    ]


def setup_workspace_ir_abs(ws):
    # Absorption lookup table
    ws.abs_nls_interp_order = 3
    ws.abs_t_interp_order = 3
    ws.abs_p_interp_order = 3
    path_abs_lookup_table = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data/lookup_tables/abs_table_Earthcare_TIR2_one_wavelength_2026-02-11_15:22:46.062825.xml",
    )
    ws.ReadXML(
        ws.abs_lookup,
        path_abs_lookup_table,
    )
    ws.abs_species = ws.abs_lookup.value.species

    # Sensor
    wsv.sensor_poslos(ws, z=[400e3], za=[180])
    ea.sensorNone(
        ws,
        f_grid=ws.abs_lookup.value.f_grid.value,
        iy_unit=IyUnit.PLANCK_BT,
    )

    # Compute absorption agenda
    ws.abs_lines_per_speciesSetEmpty()
    ws.propmat_clearsky_agendaAuto(use_abs_lookup=1)
    ws.abs_lookupAdapt()


def setup_workspace_aws(ws):
    # Absorption
    ea.abs_speciesPredefined(
        ws, add_lwc=False, option=AbsSpeciesPredefinedOption.RTTOV_v13x
    )

    # Sensor
    from handy import AWSChannel

    # check channels are stricktly sorted and have no duplicates
    channels = [AWSChannel.AWS33, AWSChannel.AWS36, AWSChannel.AWS41, AWSChannel.AWS44]
    ws.iy_unit = IyUnit.PLANCK_BT.value
    # check that channels are strictly sorted
    nch = len(channels)
    for i in range(nch - 1):
        if channels[i + 1].value <= channels[i].value:
            raise ValueError("fsetting.channels must be sorted and have no duplicates.")
    # create sensor reponse matrix
    aws_specs = sensor.aws.sensor_specs(
        channels=[ch.name for ch in channels],
    )
    aws_specs.populate_with_srf_delta()
    # assemble f_grid
    f_grid = aws_specs.fgrid_from_srf(equidistant=True, n_weight=1)
    ws.f_grid = f_grid
    # assemble f_backend
    f_backend = np.empty(nch)
    for idx_ch, ch in enumerate(channels):
        spec = aws_specs.channels[ch.name]
        # Note: For AWS4X we use the upper band, otherwise the centre frequency
        f_backend[idx_ch] = spec.f_centre + spec.f_offset[-1]
    ws.f_backend = f_backend
    # get channel2fgrid_indexes/weights
    hvecs = hmatrix.frequency_weights(aws_specs, f_grid)
    channel2fgrid_indexes = [None] * nch
    channel2fgrid_weights = [None] * nch
    for i in range(nch):
        if len(hvecs[i]) == 1:
            hthis = hvecs[i][0]
        else:
            hthis = hvecs[i][0] + hvecs[i][1]
        ind = np.nonzero(hthis)[0]
        channel2fgrid_indexes[i] = ind
        channel2fgrid_weights[i] = hthis[ind]
    # create sensor response
    ws.channel2fgrid_indexes = channel2fgrid_indexes
    ws.channel2fgrid_weights = channel2fgrid_weights
    ws.FlagOn(ws.sensor_norm)
    ws.AntennaOff()
    ws.sensor_responseInit()
    ws.sensor_responseMixerBackendPrecalcWeights()

    wsv.sensor_poslos(ws, z=[600e3], za=[170])


def setup_ir(ws, dset, pmodels_ir, pr0, prn, skip):
    """Setup IR simulation and process profiles."""
    print("Setting up IR simulation...")
    print(f"Using particle model for FWC: {pmodels_ir[1].habit_name}")

    # Fixed parts of cloudbox
    pm.to_scat_data(ws, pmodels_ir)
    ea.cloudbox(ws, CloudboxOption.FULL)

    n = len(range(pr0, prn, skip))
    print(f"Processing {n} IR profiles from {pr0} to {prn} with skip {skip}.")

    y_allsky_ir = np.full((n, len(ws.f_grid.value)), np.nan)
    for i, pr in tqdm(
        enumerate(range(pr0, prn, skip)), total=n, desc="Processing IR Profiles"
    ):
        ok = dl.ec_xmetAtmosphere1D(ws, dset, pr, FascodVersion.MLS)
        if ok:
            # Additional processing can be done here if needed
            psd_input = dl.ec_acm_capHydrometeors1D(
                ws.p_grid.value,
                dset,
                pr,
            )
            pm.to_pnd_field_1d(ws, pmodels_ir, psd_input)
            ea.checks(ws)
            ea.disort(ws, Npfct=-1)
            ws.yCalc()
            y_allsky_ir[i, :] = ws.y.value

    print("IR all-sky TBs computed successfully.")

    result_ir = xr.DataArray(
        data=np.atleast_2d(y_allsky_ir),
        dims=["along_track", "frequency_ir"],
        coords={
            "along_track": np.arange(pr0, prn, skip),
            "frequency_ir": ws.f_grid.value.value.copy(),
        },
        name="ARTS_MSI_brightness_temperature",
        attrs={
            "units": "K",
            "long_name": "ARTS simulated MSI brightness temperature",
            "description": "ARTS simulated MSI brightness temperature using ACMCAP hydrometeors",
        },
    )

    result_ir = result_ir.expand_dims(("psd", "habit_ir")).assign_coords(
        {
            "psd": [pmodels_ir[1].psd.value],
            "habit_ir": [pmodels_ir[1].habit_name],
        }
    )
    return result_ir


def setup_aws(ws, dset, pmodels_mw, pr0, prn, skip):
    """Setup AWS simulation and process profiles."""
    print("Setting up AWS simulation...")
    print(f"Using particle model for FWC: {pmodels_mw[1].habit_name}")

    # Fixed parts of cloudbox
    pm.to_scat_data(ws, pmodels_mw)
    ea.cloudbox(ws, CloudboxOption.FULL)

    n = len(range(pr0, prn, skip))
    print(f"Processing {n} AWS profiles from {pr0} to {prn} with skip {skip}.")

    y_allsky_mw = np.full((n, len(ws.f_backend.value)), np.nan)
    for i, pr in tqdm(
        enumerate(range(pr0, prn, skip)), total=n, desc="Processing AWS Profiles"
    ):
        ok = dl.ec_xmetAtmosphere1D(ws, dset, pr, FascodVersion.MLS)
        if ok and dset["land_flag"].isel(along_track=pr).values == 0:
            psd_input = dl.ec_acm_capHydrometeors1D(
                ws.p_grid.value,
                dset,
                pr,
            )
            pm.to_pnd_field_1d(ws, pmodels_mw, psd_input)
            ea.checks(ws)
            ea.disort(ws, Npfct=-1)
            ws.yCalc()
            y_allsky_mw[i, :] = ws.y.value

    print("AWS all-sky TBs computed successfully.")

    result_aws = xr.DataArray(
        data=np.atleast_2d(y_allsky_mw),
        dims=["along_track", "frequency_aws"],
        coords={
            "along_track": np.arange(pr0, prn, skip),
            "frequency_aws": ws.f_backend.value.value.copy(),
        },
        name="ARTS_AWS_brightness_temperature",
        attrs={
            "units": "K",
            "long_name": "ARTS simulated AWS brightness temperature",
            "description": "ARTS simulated AWS brightness temperature using ACMCAP hydrometeors",
        },
    )
    result_aws = result_aws.expand_dims(("psd", "habit_mw")).assign_coords(
        {
            "psd": [pmodels_mw[1].psd.value],
            "habit_mw": [pmodels_mw[1].habit_name],
        }
    )
    
    return result_aws


def merge_results(
    result_ir, result_aws, dset, ds_xmet, dset_fmr, orbit, frame, pr0, prn, skip
):
    """Merge IR and AWS results into a single dataset."""
    print("Merging IR and AWS results...")

    result = xr.merge([result_ir, result_aws], combine_attrs="drop_conflicts")
    result = result.assign_attrs(
        {
            "description": "ARTS simulated brightness temperatures using ACMCAP hydrometeors",
            "orbit": orbit,
            "frame": frame,
            "ACMCAP_source": dset.encoding["source"].split("/")[-1],
            "XMET_source": ds_xmet.encoding["source"].split("/")[-1],
            "CFMR_source": dset_fmr.encoding["source"].split("/")[-1],
        }
    )
    result = result.assign(
        {
            "latitude": (
                "along_track",
                dset["latitude"].isel(along_track=slice(pr0, prn, skip)).values,
            ),
            "longitude": (
                "along_track",
                dset["longitude"].isel(along_track=slice(pr0, prn, skip)).values,
            ),
            "time": (
                "along_track",
                dset["time"].isel(along_track=slice(pr0, prn, skip)).values,
            ),
            "land_flag": (
                "along_track",
                dset["land_flag"].isel(along_track=slice(pr0, prn, skip)).values,
            ),
        }
    )

    return result


def plot_results(dset, result, pr0, prn, skip):
    """Create visualization plots of simulation results."""
    print("Generating plots...")

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
    ax[0].set_title("Rain Water Content")
    ax[0].set_xlabel("")

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
    ax[1].set_title("Liquid Water Content")
    ax[1].set_xlabel("")

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
    ax[2].set_title("Ice Water Content")
    ax[2].set_xlabel("")

    # Brightness temperatures
    dset["MSI_longwave_brightness_temperature"].isel(
        along_track=slice(pr0, prn, skip), MSI_longwave_channel=1
    ).plot(ax=ax[3], marker="x", alpha=0.7, ls="-", color="black", label="MSI 10.8um")
    result["ARTS_MSI_brightness_temperature"].plot(
        ax=ax[3], hue="frequency_ir", marker="x", alpha=0.7, ls="-", label="ARTS 10.8um"
    )
    result["ARTS_AWS_brightness_temperature"].plot(
        ax=ax[3], hue="frequency_aws", marker="o", alpha=0.7, ls="-", label="ARTS AWS"
    )
    ax[3].set_title("Brightness Temperatures")
    ax[3].set_xlabel("Along Track Index")
    ax[3].set_ylabel("Brightness Temperature (K)")
    handles, _ = ax[3].get_legend_handles_labels()
    new_labels = ["MSI 10.8 um", "ARTS 10.8 um"]
    for freq in result.frequency_aws.values:
        new_labels.append(f"ARTS {freq/1e9:.1f} GHz")
    ax[3].legend(handles, new_labels, loc="upper right")
    plt.show()


# %%
# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main(
    orbit="06203",
    frame="D",
    pr0=None,
    prn=None,
    skip=None,
    plot=False,
    output=None,
    ice_habit_ir=["8-ColumnAggregate-ModeratelyRough"],
    ice_habit_aws=["LargePlateAggregate"],
):
    """Main execution function.

    Parameters
    ----------
    orbit : str
        Orbit number (default: 06203)
    frame : str
        Frame identifier (default: D)
    pr0 : int
        First profile index (default: 0)
    prn : int
        Last profile index (default: max available in dataset along_track dimension)
    skip : int
        Skip profiles (default: 1)
    plot : bool
        Show plots (default: False)
    output : str
        Output NetCDF file path (optional)
    ice_habit_ir : list of str
        Ice habit(s) for IR simulation (default: ["8-ColumnAggregate-ModeratelyRough"])
    ice_habit_aws : list of str
        Ice habit(s) for AWS simulation (default: ["LargePlateAggregate"])
    """
    print("=" * 70)
    print(f"ARTS Simulation: orbit={orbit}, frame={frame}")
    print(f"Profile range: {pr0}-{prn} (skip={skip})")
    print(f"Plotting: {plot}, Output: {output}")

    print("=" * 70)

    # % Load data
    dset, ds_xmet, dset_fmr = load_data(orbit, frame)
    if pr0 is None:
        pr0 = 0
    if prn is None:
        prn = dset.sizes["along_track"]
    if skip is None:
        skip = 1

    # % Initialize workspace
    ws = init_workspace()

    # % Setup and run IR simulation
    setup_workspace_ir_abs(ws)

    if isinstance(ice_habit_ir, str):
        ice_habit_ir = [ice_habit_ir]
    result_ir = []
    for habit in ice_habit_ir:
        print(f"IR ice habit: {habit}")
        pmodels_ir = get_default_pmodels_ir(ice_habit=habit)
        result_ir_habit = setup_ir(ws, dset, pmodels_ir, pr0, prn, skip)
        result_ir.append(result_ir_habit)
    result_ir = xr.concat(result_ir, dim="habit_ir")

    # % Setup and run AWS simulation
    setup_workspace_aws(ws)

    if isinstance(ice_habit_aws, str):
        ice_habit_aws = [ice_habit_aws]
    result_aws = []
    for habit in ice_habit_aws:
        print(f"AWS ice habit: {habit}")
        pmodels_aws = get_default_pmodels_aws(ice_habit=habit)
        result_aws_habit = setup_aws(ws, dset, pmodels_aws, pr0, prn, skip)
        result_aws.append(result_aws_habit)
    result_aws = xr.concat(result_aws, dim="habit_mw")

    # % Merge results
    result = merge_results(
        result_ir, result_aws, dset, ds_xmet, dset_fmr, orbit, frame, pr0, prn, skip
    )

    # % Save if output file specified
    if output:
        print(f"Saving results...")
        result.to_netcdf(output)
        print(f"Results saved to {output}")

    # %Plot if requested
    if plot:
        plot_results(dset, result, pr0, prn, skip)

    print("=" * 70)
    print("Simulation completed successfully!")
    print("=" * 70)

    return result, dset


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ARTS Simulation for AWS and IR Brightness Temperatures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default orbit and show plots
  python cap2tb_aws_ir.py --plot
  
  # Run specific orbit with custom profile range
  python cap2tb_aws_ir.py --orbit 06250 --frame D --pr0 1000 --prn 2000 --skip 50
  
  # Run and save to file without plotting
  python cap2tb_aws_ir.py --orbit 06203 --frame D --output results.nc --no-plot
        """,
    )

    parser.add_argument(
        "--orbit", type=str, default="06203", help="Orbit number (default: 06203)"
    )
    parser.add_argument(
        "--frame", type=str, default="D", help="Frame identifier (default: D)"
    )
    parser.add_argument(
        "--pr0", type=int, default=None, help="First profile index (default: 0)"
    )
    parser.add_argument(
        "--prn",
        type=int,
        default=None,
        help="Last profile index (default: max available in dataset along_track dimension)",
    )
    parser.add_argument(
        "--skip", type=int, default=1, help="Skip profiles (default: 1)"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Show plots (default: no plots)"
    )
    parser.add_argument(
        "--no-plot", action="store_false", dest="plot", help="Do not show plots"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output NetCDF file path (optional)"
    )
    parser.add_argument(
        "--ice-habit-ir",
        type=str,
        default="Plate-ModeratelyRough",
        help="Ice habit for IR simulation (default: Plate-ModeratelyRough)",
    )
    parser.add_argument(
        "--ice-habit-aws",
        type=str,
        default="LargePlateAggregate",
        help="Ice habit for AWS simulation (default: LargePlateAggregate)",
    )

    args = parser.parse_args()

    result, dset = main(
        orbit=args.orbit,
        frame=args.frame,
        pr0=args.pr0,
        prn=args.prn,
        skip=args.skip,
        plot=args.plot,
        output=args.output,
        ice_habit_ir=args.ice_habit_ir.split(","),
        ice_habit_aws=args.ice_habit_aws.split(","),
    )
