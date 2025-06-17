# IR simulation helper functions

import easy_arts.wsv as wsv
from easy_arts.data_model import FascodVersion, SpectralRegion
import easy_arts.easy_arts as ea
import numpy as np
import pyarts
import datetime
import os
import time


def create_and_save_abs_lookup(
    f_grid,
    p_grid,
    abs_species: list,
    t_perturbations: np.ndarray = np.linspace(-150, 50, num=4),
    h2o_perturbations: np.ndarray = np.array([0.05, 0.25, 1, 4]),
    datapath: str = os.path.join(os.getcwd(), "data"),
    filename: str = "lookup_table.xml",
):
    """
    Calculate the lookup table and save to a file

    """

    # Initiate workspace basics
    ws = ea.wsInit(SpectralRegion.TIR)
    ea.atmosphereDim(ws, dim=1)
    ws.stokes_dim = 1
    ws.abs_speciesSet(species=abs_species)

    # Download ARTS catalogs if they are not already present.
    pyarts.cat.download.retrieve()
    ws.ReadXML(ws.predefined_model_data, "model/mt_ckd_4.0/H2O.xml")  # temp
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)

    ws.lbl_checkedCalc()  # Check that the line-by-line data is consistent
    ws.propmat_clearsky_agendaAuto()

    # Create lookup table
    wsv.abs_lookupFascod(
        ws,
        fascod_atm=FascodVersion.TRO,
        f_grid=f_grid,
        p_grid=p_grid,
        t_perturbations=t_perturbations,
        h2o_nls=True,
        o2_nls=False,
        nls_perturbations=h2o_perturbations,
    )

    # Write into an XML file
    ws.output_file_formatSetBinary()
    ws.WriteXML(
        input=ws.abs_lookup,
        filename=os.path.join(datapath, filename),
    )


if __name__ == "__main__":

    current_time = datetime.datetime.now()

    h2o_perturbations = np.array([0.01, 0.05, 0.25, 0.5, 0.7, 0.9, 1, 1.1, 1.3, 1.8, 2, 4, 10, 16])
    t_perturbations = np.linspace(-150, 50, num=8)

    # f_grid_kayser = np.linspace(800, 950, 500)  # Kayser cm-1
    # f_grid = pyarts.arts.convert.kaycm2freq(f_grid_kayser)  # Convert to Hz
    wavelen_grid = np.linspace(11.25e-6, 10.35e-6, 50)  # Wavelength in meters in descending order
    f_grid = pyarts.arts.convert.wavelen2freq(wavelen_grid)  # Convert to Hz
    p_grid = np.logspace(np.log10(1050e2), np.log10(10e2), 150)  # Pa
    abs_species = [
        "H2O, H2O-SelfContCKDMT400 , H2O- ForeignContCKDMT400",
        # "O3",
        "CO2, CO2-CKDMT252",
        # "CH4",
        "N2O",
    ]
    start = time.time()
    create_and_save_abs_lookup(
        f_grid=f_grid,
        p_grid=p_grid,
        abs_species=abs_species,
        t_perturbations=t_perturbations,
        h2o_perturbations=h2o_perturbations,
        datapath='/home/anqil/arts_ir_simulation/data/lookup_tables',
        filename="abs_table_{}_{}.xml".format(
            "Earthcare_TIR2",
            current_time,
        ).replace(" ", "_"),
    )
    end = time.time()
    print(end - start, "s")

