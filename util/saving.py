import os
import json
from typing import Dict

import torch

def model_saving(file_path: str, model_dict, checkpoint_num: int):
    torch.save(
        model_dict,
        os.path.join(file_path, 'checkpoint_{}.pth'.format(checkpoint_num))
    )


def json_saving(file_path: str, json_data: Dict):
    with open(file_path, "w", encoding="utf8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
