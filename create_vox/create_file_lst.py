import json
import os
import platform
CUR_PATH = os.path.dirname(os.path.realpath(__file__))

version = platform.version()[-1]
if version == "7":
    infofile = "vv_info.json"
else:
    infofile = "info.json"
print("infofile:", infofile)

def get_all_info():
    with open(CUR_PATH + '/' + infofile) as json_file:
        data = json.load(json_file)
        lst_dir, cats, all_cats, raw_dirs = data["lst_dir"], data['cats'], data['all_cats'], data["raw_dirs_v1"]
    return lst_dir, cats, all_cats, raw_dirs

if __name__ == "__main__":

    # nohup python -u create_file_lst.py &> create_imgh5.log &

    lst_dir, cats, all_cats, raw_dirs = get_all_info()
