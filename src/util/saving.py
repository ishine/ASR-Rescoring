import os
from typing import Dict

import torch

def model_saving(file_path: str, model_dict, checkpoint_num: int):
    torch.save(
        model_dict,
        os.path.join(file_path, 'checkpoint_{}.pth'.format(checkpoint_num))
    )

def loss_saving(file_path: str, loss: Dict):
    for loss_type, loss_record in loss.items():
        with open(file_path + "loss.txt", "w") as f:
            f.write(f"{loss_type} loss: \n")
            f.write(" ".join([str(loss) for loss in loss_record]) + "\n")
