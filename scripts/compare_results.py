# %%
import enum
import os
import xarray as xr
import matplotlib.pyplot as plt

# %% read datasets and compare
ds = xr.open_mfdataset(os.path.join(os.getcwd(), "data/lookup_table_orders/y_perturb14_order3*.nc"))
da = ds.to_array()  # .reindex(variable=["line_by_line", "perturb4_order3", "perturb14_order3", "perturb14_order7"])

fig, axs = plt.subplots(2, 1, sharex=True)
da.plot(ax=axs[0], hue="variable")
da_diff = da.drop_sel(variable="line_by_line") - da.sel(variable="line_by_line")
da_diff.plot(ax=axs[1], hue="variable")
axs[0].set_ylabel("K")
axs[0].set_xlabel("")
axs[0].set_title("Simulated Brightness Temperature")
axs[1].set_ylabel("K")
axs[1].set_xlabel("cm-1")
axs[1].set_title("Difference to line_by_line")
plt.show()

# %% compare all sky spectra with different particle habits and PSDs
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(10, 6))

ds_DelanoeEtAl14 = xr.open_mfdataset(
    os.path.join(
        os.getcwd(),
        "data/allsky_habits_DelanoeEtAl14/*nc",
    ),
)
ds_DelanoeEtAl14["f_grid"] = ds_DelanoeEtAl14.f_grid.assign_attrs({"unit": "cm-2"})
ds_DelanoeEtAl14.to_array().assign_attrs({"unit": "K"}).plot(
    hue="variable",
    marker=".",
    ax=axes[0],
)

ds_FieldEtAl07 = xr.open_mfdataset(
    os.path.join(os.getcwd(), "data/allsky_habits_FieldEtAl07/*nc"),
)
ds_FieldEtAl07["f_grid"] = ds_FieldEtAl07.f_grid.assign_attrs({"unit": "cm-2"})
ds_FieldEtAl07.to_array().assign_attrs({"unit": "K"}).plot(
    hue="variable",
    marker=".",
    ax=axes[1],
)

ds_MgdMass = xr.open_mfdataset(
    os.path.join(os.getcwd(), "data/allsky_habits_MgdMass/*nc"),
)
ds_MgdMass["f_grid"] = ds_MgdMass.f_grid.assign_attrs({"unit": "cm-2"})
ds_MgdMass.to_array().assign_attrs({"unit": "K"}).plot(
    hue="variable",
    marker=".",
    ax=axes[2],
)

axes[0].set_title("DelanoeEtAl14")
axes[1].set_title("FieldEtAl07")
axes[2].set_title("MgdMass (default)")

# %% compare subsampling on clear sky spectra
ds = xr.open_dataset(
    os.path.join(
        os.getcwd(),
        "data/lookup_table_orders/y_line_by_line.nc",
    ),
)
ds["f_grid"] = ds.f_grid.assign_attrs({"unit": "cm-2"})
da = ds.to_array().assign_attrs({"unit": "K"})
da.plot(marker=".", label="original")

da.sel(f_grid=slice(None, None, 10)).plot(
    marker=".",
    label="subsampling:10",
)
# plt.legend()

# %%
file_list = os.listdir(
    os.path.join(
        os.getcwd(),
        "data/allsky_bulkprop",
    )
)


def get_label(ds, labels):
    str_label = {}
    for label in labels:
        str_label[label] = ds["Droxtal-Smooth"].attrs[label].item()
    return str_label


for i in range(len(file_list)):
    ds = xr.open_dataset(
        os.path.join(
            os.getcwd(),
            "data/allsky_bulkprop",
            file_list[i],
        )
    )
    ds.to_array().plot(
        marker=".",
        label=get_label(ds, ["iwp", "z_ref", "fwhm"]),
    )
plt.legend(bbox_to_anchor=(0.25, 0.4), loc="upper left")
plt.ylabel("K")
plt.xlabel("cm-2")

# %%
