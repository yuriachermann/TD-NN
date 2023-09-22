import numpy as np
from matplotlib import pyplot as plt
import cv2

def smooth(f, K=5):
    """Smoothing a function using a low-pass filter (mean) of size K"""
    kernel = np.ones(K) / K
    f = np.concatenate([f[: int(K // 2)], f, f[int(-K // 2) :]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K // 2 : -K // 2]  # removing boundary-fixes
    return smooth_f


def plot_results(train_result, train_eval_result, labels=["", ""]):
    """
    Plots a graph side by side, used for train and eval graphs (either accuracy or loss).
    Mode sets the strings for the graph plots
    """
    plt.style.use("seaborn")
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))

    train_result = np.array(train_result)
    ax[0].plot(train_result, c="blue", label=labels[0], linewidth=3, alpha=0.5)
    ax[0].plot(smooth(train_result, 10), c="cornflowerblue", label=f"Smoothed {labels[0]}", linewidth=3, alpha=0.5)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_title(labels[0])

    # since our evaluation loss is a nested list
    train_eval_result = np.array(train_eval_result)
    ax[1].plot(train_eval_result, c="red", label=labels[1], linewidth=3, alpha=0.5)
    ax[1].plot(
        smooth(train_eval_result, 10), c="lightcoral", label=f"Smoothed {labels[1]}", linewidth=3, alpha=0.5
    )
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Iteration")
    ax[1].set_title(labels[1])

    ax[2].plot(train_result, c="blue", label=labels[0], linewidth=3, alpha=0.5)
    ax[2].plot(smooth(train_result, 10), c="cornflowerblue", label=f"Smoothed {labels[0]}", linewidth=3, alpha=0.5)
    ax[2].plot(train_eval_result, c="red", label=labels[1], linewidth=3, alpha=0.5)
    ax[2].plot(
        smooth(train_eval_result, 10), c="lightcoral", label=f"Smoothed {labels[1]}", linewidth=3, alpha=0.5
    )
    ax[2].legend(loc="best")
    ax[2].set_xlabel("Iteration")
    ax[2].set_title(f"{labels[0]} and {labels[1]}")

    return fig, ax

def segment_mask_as_overlay(img, mask, color):

    match color:
        case "blue":
            rgb_color = [57 / 255, 57 / 255, 254 / 255]
        case "red":
            rgb_color = [254 / 255, 57 / 255, 57 / 255]
        case _:
            assert color in ["red", "blue"], print("Color so far must be red or blue.")

    seg_mask = cv2.convertScaleAbs(mask)
    rgb_seg = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB).astype(np.float32)
    coloring_mask = (rgb_seg == [1.0, 1.0, 1.0]).all(axis=2)
    rgb_seg[coloring_mask] = rgb_color

    vb_img = img.copy()
    vb_img[seg_mask == 1.0, :] *= 0.3
    vb_img = cv2.addWeighted(vb_img, 1, rgb_seg, 0.5, 0, dtype=cv2.CV_32F)
    return vb_img