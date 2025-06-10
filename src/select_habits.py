# %%
import os


# %%
def get_habit_list_from_stdhabits():
    file_list = os.listdir(os.environ["PATH_STDHABITS"])
    return [f.removesuffix(".xml.bin") for f in file_list if ".xml.bin" in f]


def get_habit_list_from_path(
    path: str,
    prefix: str = "",
    suffix: str = "",
):
    file_list = os.listdir(path)
    return [f.removeprefix(prefix).removesuffix(suffix) for f in file_list if suffix in f]


if __name__ == "__main__":
    print(get_habit_list_from_stdhabits())

    # print(
    #     get_habit_list_from_path(
    #         "/home/anqil/arts_ir_simulation/data/allsky_habits",
    #         "y_",
    #         ".nc",
    #     )
    # )
# %%
