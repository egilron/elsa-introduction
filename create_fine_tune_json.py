import os, json
from pathlib import Path
from datetime import datetime
timestamp = datetime.now().strftime("%m%d%H%M")

# Create the json files with model args, data args and training args for the experiments.
# First, set a default
# Then provide the lists of options
# Then provide any manual additional settings

CONFIG_ROOT = "configs"

# Use same keys here as in ds:
label_col = {"old_tsa": "tsa_tags",
    "tsa": "tsa_tags",
    "ner": "ner_tags", 
    "elsa": "elsapol_tags"}

# Settings dependent on where we are working
ms, ds, local_out_dir = None, None, None
WHERE = "msi"


# Lumi datapaths
# ds = {"old_tsa": "/users/rnningst/sequence-labelling/data/tsa_arrow", 
#     "tsa": "/users/rnningst/sequence-labelling/data/tsa_arrow", 
#       "ner": "/users/rnningst/sequence-labelling/data/ner_arrow"
#     }
# ds = {"tsa": "/users/rnningst/sequence-labelling/data/tsa_arrow"
#     }



if WHERE == "fox":
    ms = {  "nb-bert-base": "NbAiLab/nb-bert-base" ,
            "xlm-roberta-base":"/fp/projects01/ec30/models/xlm-roberta-base"}
    local_out_dir = "/fp/projects01/ec30/egilron/seq-label"
    ds = {"elsa": None
        }

# HP
if WHERE == "hp":
    ms = {  # Point to local cache instead
            "nb-bert-base": "nb-bert-base" ,
            "xlm-roberta-base":"xlm-roberta-base"}
    ds = {"elsa": "data/ds_elsa"}
    local_out_dir = "output"

# HP
if WHERE == "msi":
    ms = {  # Or point to local cache instead
            "nb-bert-base": "nb-bert-base" ,
            "xlm-roberta-base":"xlm-roberta-base"}
    ds = {"elsa": "data/ds_elsa"}
    local_out_dir = "output"

assert not any([e is None for e in [ms, ds, local_out_dir]]), "ms, ds, and local_out_dir need values set above here"
# Add training args as needed
default = {
    "model_name_or_path": None, #ms["brent0"] ,
    "dataset_name": None,
    "per_device_train_batch_size": 32,
    "task_name": "elsa_sq-l", # Change this in iteration
    "output_dir": local_out_dir, #"/project/project_465000144/sq_label", # Add to this in iteration
    "overwrite_cache": True,
    "overwrite_output_dir": True,
    "do_train": True,
    "num_train_epochs": 10,
    "do_eval": True,
    "return_entity_level_metrics": False, # True,
    "use_auth_token": False,
    "logging_strategy": "epoch",
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch", #"epoch",
    "save_total_limit": 4,
    "load_best_model_at_end": True, # True,
    "label_column_name": None,
    "disable_tqdm": True,
    "do_predict": True,
    "text_column_name": "tokens"
}


# Iterations: design this according to needs
for task in ds.keys():
    experiments = [] # List of dicts, one dict per experiments: Saves one separate json file each
    for m_name, m_path in ms.items():
        for sd in [101]:#,202,303,404,505]:
            exp = default.copy()
            exp["model_name_or_path"] = m_path
            exp["dataset_name"] = ds [task]
            exp["task_name"] = f"{timestamp}-{sd}_{task}_{m_name}"
            exp["output_dir"] = os.path.join(default["output_dir"], exp["task_name"] )
            exp["label_column_name"] = label_col.get(task, "")
            exp["seed"] = sd

            experiments.append(exp)

    for i, exp in enumerate(experiments): # Move this with the experiments list definition to make subfolders
        # if  i== 0 or not all ([exp[key]== default.get(key) for key in exp.keys()]):
        print(exp["task_name"])
        save_path = Path(CONFIG_ROOT, task, exp["task_name"]+".json")
        save_path.parent.mkdir( parents=True, exist_ok=True)
        with open(save_path, "w") as wf:
            json.dump(exp, wf)





