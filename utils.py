import matplotlib.pyplot as plt
import numpy as np


def show_message(text, verbose=True, end='\n'):
    if verbose: print(text, end=end)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_tensor_to_numpy(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="bottom", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data
