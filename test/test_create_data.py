import pandas as pd
import os

# create test data
INP_FINA = "link-CJ48.92-Flow_velocity-df.pickle"
INP_COLNA = "link/CJ48.92/Flow_velocity"
OUT_FIPA = "example_data/link-CJ48.92-Flow_velocity-df.csv"
BASE_FDPA = "/mnt/d/andre/data_all/DB_proj_2DmodelingSpeedupProb/04_extracted_simulations/sm01_train-validate/outputs"
ALL_FXPA = [os.path.join(BASE_FDPA, fxpa) for fxpa in os.listdir(BASE_FDPA)]
ALL_FDPA = [fxpa for fxpa in ALL_FXPA if os.path.isdir(fxpa)]
del ALL_FXPA, BASE_FDPA
all_data = {}
common_index = None
for cur_fdpa in ALL_FDPA:
    cur_fipa = os.path.join(cur_fdpa, INP_FINA)
    if not os.path.exists(cur_fipa):
        del cur_fdpa, cur_fipa
        continue
    cur_fdna = os.path.split(cur_fdpa)[-1]
    cur_sr = pd.read_pickle(cur_fipa)
    common_index = list(range(cur_sr.size)) if common_index is None else common_index
    # cur_sr.set_index() = cur_sr.reindex(common_index)
    cur_sr = cur_sr.reset_index()[INP_COLNA]
    all_data[cur_fdna] = cur_sr
    print("Read {0} into {1}".format(type(cur_sr), cur_fdna))
    del cur_fdpa, cur_fipa, cur_sr
out_df = pd.DataFrame(data=all_data)
out_df.to_csv(OUT_FIPA)
print("Wrote: %s", OUT_FIPA)
