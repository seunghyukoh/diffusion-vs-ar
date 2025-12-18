import json
import math
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend
from typing import List, Optional

import matplotlib.pyplot as plt
from transformers.trainer import TRAINER_STATE_NAME

from llmtuner.extras.logging import get_logger

logger = get_logger(__name__)


def smooth(scalars: List[float]) -> List[float]:
    r"""
    EMA implementation according to TensorBoard.
    """
    last = scalars[0]
    smoothed = list()
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_loss(save_dictionary: os.PathLike, keys: Optional[List[str]] = None) -> None:

    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), "r", encoding="utf-8") as f:
        data = json.load(f)
    if keys is None:
        keys = data["log_history"][0].keys()
    for key in keys:
        steps, metrics = [], []
        for i in range(len(data["log_history"])):
            if key in data["log_history"][i]:
                steps.append(data["log_history"][i]["step"])
                metrics.append(data["log_history"][i][key])

        if len(metrics) == 0:
            logger.warning(f"No metric {key} to plot.")
            continue

        # Convert to numpy arrays for better compatibility
        steps = np.array(steps, dtype=np.float64)
        metrics = np.array(metrics, dtype=np.float64)

        plt.figure()
        plt.plot(steps, metrics, alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics.tolist()), label="smoothed")
        plt.title("training {} of {}".format(key, save_dictionary))
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        plt.savefig(
            os.path.join(save_dictionary, "training_{}.png".format(key)), format="png", dpi=100
        )
        plt.close()  # Close the figure to free memory and avoid conflicts
        print("Figure saved:", os.path.join(save_dictionary, "training_{}.png".format(key)))
