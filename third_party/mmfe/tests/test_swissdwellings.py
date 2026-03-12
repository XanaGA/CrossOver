import os
import numpy as np
import matplotlib.pyplot as plt
from third_parties.SwissDwellings.constants import ROOM_NAMES, CMAP_ROOMTYPE
from third_parties.SwissDwellings.utils import colorize_floorplan


if __name__ == "__main__":
    # set data path and child paths
    datapath = 'data/SwissDwellings/modified-swiss-dwellings-v2/train'

    path = {
        "full": datapath,
        "graph_in": os.path.join(datapath, 'graph_in'),
        "struct_in": os.path.join(datapath, 'struct_in'),
        "full_out": os.path.join(datapath, 'full_out'),
        "graph_out": os.path.join(datapath, 'graph_out')
    }

    ids = [ 23, 322, 630, 706, 832, 972, 1031, 1323]

    # set up figure
    fs = 10
    fig, axs = plt.subplots(2, 4, figsize=(fs*4, fs*2))
    axs = axs.flatten()

    for i, id in enumerate(ids):

        # set axis
        ax = axs[i]
        _ = [ax.axis('off'), ax.axes.set_aspect('equal')]

        # get structural components
        stack = np.load(os.path.join(path["struct_in"], f'{id}.npy'))

        # channel 1: structural components
        # note: channel 2 and 3 are x and y locations
        #   this holds for "full_out" as well
        struct = stack[..., 0].astype(np.uint8)
        ax.imshow(struct, cmap='gray')
    plt.show()




    class_mapping = {cat: index for index, cat in enumerate(ROOM_NAMES)}
    CLASSES = list(map(class_mapping.get, ROOM_NAMES))

    # set up figure
    fs = 10
    fig, axs = plt.subplots(2, 4, figsize=(fs*4, fs*2))
    axs = axs.flatten()

    for i, id in enumerate(ids):

        # set axis
        ax = axs[i]
        _ = [ax.axis('off'), ax.axes.set_aspect('equal')]

        # get structural components
        stack = np.load(os.path.join(path["full_out"], f'{id}.npy'))

        # channel 1: structural components
        # note: channel 2 and 3 are x and y locations
        #   this holds for "full_out" as well
        img = stack[..., 0].astype(np.uint8)
        ax.imshow(colorize_floorplan(img, classes=CLASSES, cmap=CMAP_ROOMTYPE))

    plt.show()



