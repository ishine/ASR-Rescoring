import sys
sys.path.append("/home/chkuo/chkuo/experiment/ASR-Rescoring/src")
from util.plot import plot
import json

json_data = json.load(
                open(
                    "/home/chkuo/chkuo/experiment/ASR-Rescoring/result/Fohr21INTERSPEECH/alsem/loss.json",
                    "r",
                    encoding="utf-8"
                )
            )

plot(
    "/home/chkuo/chkuo/experiment/ASR-Rescoring/result/Fohr21INTERSPEECH/alsem/every_epoch_loss",
    ["train", "dev"],
    [json_data["train"]["every_epoch"], json_data["dev"]["every_epoch"]],
    {"x": "epoch", "y": "loss"}
)

plot(
    "/home/chkuo/chkuo/experiment/ASR-Rescoring/result/Fohr21INTERSPEECH/alsem/every_batch_loss",
    ["train", "dev"],
    [json_data["train"]["every_1000_batch"], json_data["dev"]["every_1000_batch"]],
    {"x": "every 1000 batch", "y": "loss"}
)