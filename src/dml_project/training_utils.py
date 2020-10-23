import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, Union, List

LOSS_KEYS = [
    "loss",
    "loss_classifier",
    "loss_box_reg",
    "loss_objectness",
    "loss_rpn_box_reg",
]


def setup_plots(n_metrics=5):
    plt.ion()

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    axs_dict = {
        LOSS_KEYS[0]: axs[0][0],
        LOSS_KEYS[1]: axs[0][1],
        LOSS_KEYS[2]: axs[0][2],
        LOSS_KEYS[3]: axs[1][0],
        LOSS_KEYS[4]: axs[1][1],
    }
    # axs_dict = {LOSS_KEYS[i]: axs[i] for i in range(n_metrics)}
    # for k, ax in axs_dict:
    #     line, = ax.plot([])
    #     line.set_label
    return fig, axs_dict


def update_plots(fig, axs_dict: Dict, train_data: Dict, val_data: Dict, n_metrics=5):
    print(f"{train_data=}")
    print(f"{val_data=}")
    for i, (k, ax) in enumerate(axs_dict.items()):

        (train_graph,) = ax.plot(train_data[k])
        (valid_graph,) = ax.plot(val_data[k])

        ax.legend((train_graph, valid_graph), ("Training", "Validation"))
        ax.set_title(k)

    plt.pause(0.01)


def update_plot_data(
    train_data: Dict[str, List],
    new_plot_train_data: Dict[str, Union[List, float]],
    val_data: Dict[str, List],
    new_plot_val_data: Dict[str, Union[List, float]],
) -> Dict[str, List]:
    for lk in LOSS_KEYS:
        train_data[lk].append(new_plot_train_data[lk])
        val_data[lk].append(new_plot_val_data[lk])

    return train_data, val_data


def validation_string(loss_summary):
    result_string = ""
    result_string += "Validation".ljust(60)

    for k, v in loss_summary.items():
        assert isinstance(v, (list, np.ndarray)) or torch.is_tensor(v)
        if torch.is_tensor(v):
            assert len(v) > 1
        avg = np.mean(v)
        med = np.median(v)
        result_string += f"{k}: {med:.4f} ({avg:.4f})  "

    return result_string
