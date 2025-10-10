# %%
import os
import pyarts.xml as xml


# %%
datafolder = "/data/s5/scattering_data/IRDbase/Yang2016/ArtsFormat"
habit = "Plate-Smooth"

# %% meta data
M = xml.load(os.path.join(datafolder, f"{habit}.meta.xml"), search_arts_path=False)
attributes = [attr for attr in dir(M[0]) if not callable(getattr(M[0], attr)) and not attr.startswith("__")]
print("Meta data attributes:", attributes)

# %% single scattering data
S = xml.load(os.path.join(datafolder, f"{habit}.xml"), search_arts_path=False)
attributes = [attr for attr in dir(S[0]) if not callable(getattr(S[0], attr)) and not attr.startswith("__")]
print("Single scattering data attributes:", attributes)


# %%
def get_attr(obj, attr_name, default=None):
    return [getattr(obj[i], attr_name, default) for i in range(len(obj))]


ext = get_attr(S, "ext_mat_data")
f_grid = get_attr(S, "f_grid")
d_veq = get_attr(M, "diameter_volume_equ")
d_max = get_attr(M, "diameter_max")
mass = get_attr(M, "mass")
assert len(d_veq) == len(ext) == len(f_grid)
