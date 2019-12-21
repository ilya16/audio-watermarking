""" Plotting tools used in the demo. """

import matplotlib.pyplot as plt
import numpy as np
from metrics import snr, nc


def plot_signals(signal, signal_w, watermark, extracted_watermark,
                 main_title='SVD watermarking of the signal',
                 frame_borders=None, x_ticks=None, x_labels=None):
    fig, ((axs1, axw1), (axs2, axw2)) = plt.subplots(2, 2, figsize=(18, 10),
                                                     gridspec_kw={'width_ratios': [3, 1]})

    axs1.plot(signal, alpha=0.7, label='original signal')
    axs1.plot(signal_w, alpha=0.7, label='signal with watermark')
    axs1.plot(signal_w - signal, alpha=0.7, label='signal differences')
    axs1.set_title(main_title, fontsize=18)
    axs1.set_xlabel('time, s', fontsize=16)
    axs1.set_ylabel('value', fontsize=16)
    axs1.legend(loc='upper right', fontsize=16)

    axs2.plot(signal_w - signal, alpha=0.8, label='signal differences', c='green')
    axs2.set_title(f'Differences between signals (close-up), SNR = {snr(signal, signal_w):.3f}', fontsize=18)
    axs2.set_xlabel('time, s', fontsize=16)
    axs2.set_ylabel('value', fontsize=16)
    axs2.legend(loc='upper right', fontsize=16)

    if x_ticks is not None:
        axs1.set_xticks(x_ticks)
        axs1.set_xticklabels(x_labels)
        axs1.tick_params(axis='both', which='major', labelsize=14)
        axs2.set_xticks(x_ticks)
        axs2.set_xticklabels(x_labels)
        axs2.tick_params(axis='both', which='major', labelsize=14)
    axs1.grid()
    axs2.grid()

    if frame_borders is not None:
        for xc in frame_borders:
            axs1.axvline(x=xc, c='black', alpha=0.7)
            axs2.axvline(x=xc, c='black', alpha=0.7)

    axw1.imshow(watermark, cmap='gray')
    axw1.set_title('Original watermark', fontsize=18)

    axw2.imshow(extracted_watermark, cmap='gray')
    axw2.set_title('Extracted watermark', fontsize=18)

    fig.tight_layout()

    plt.show()


def plot_signal_attacks(signals, watermarks, titles, figsize=(18, 20),
                        global_title='Attacks on the watermarked signal',
                        frame_borders=None, x_ticks=None, x_labels=None):
    assert len(signals) == len(watermarks) == len(titles)

    fig, axes = plt.subplots(len(signals), 2, figsize=figsize,
                             gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(global_title, fontsize=18, y=1.01)

    def _signal_plot_data(ax):
        ax.set_xlabel('time, s', fontsize=16)
        ax.set_ylabel('value', fontsize=16)
        ax.legend(loc='upper right', fontsize=16)

        if x_ticks is not None:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid()

        if frame_borders is not None:
            for xc in frame_borders:
                ax.axvline(x=xc, c='black', alpha=0.7)

    axs, axw = axes[0, :]
    axs.plot(signals[0], alpha=0.9, label='signal')
    axs.set_title(titles[0], fontsize=18)
    _signal_plot_data(axs)
    axw.imshow(watermarks[0], cmap='gray')
    axw.set_title('Original watermark', fontsize=18)
    axw.axes.get_xaxis().set_visible(False)
    axw.axes.get_yaxis().set_visible(False)

    for i in range(1, len(signals)):
        axs, axw = axes[i, :]
        axs.plot(signals[i], alpha=0.9, label='signal with watermark')
        axs.plot(signals[i] - signals[0], alpha=0.8, label='signal differences')
        axs.set_title(titles[i] + f', SNR = {snr(signals[0], signals[i]):.3f}', fontsize=18)
        _signal_plot_data(axs)
        axw.imshow(watermarks[i], cmap='gray')
        axw.set_title(f'Extracted watermark,\nNC = {nc(watermarks[0], watermarks[i]):.3f}', fontsize=18)
        axw.axes.get_xaxis().set_visible(False)
        axw.axes.get_yaxis().set_visible(False)

    fig.tight_layout()

    plt.show()


def plot_watermarks_comp(images, rows=None, cols=1, figsize=(16, 16),
                         title='Watermarks', column_names=None, row_names=None):
    if rows is None:
        rows = len(images)
    assert len(images) <= rows * cols

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    fig.suptitle(title, fontsize=20, y=1.03)

    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]

    for i in range(len(images)):
        for j in range(len(images[0])):
            ax = axes[i][j]
            ax.imshow(images[i][j], cmap='gray')

            if i == 0 and column_names is not None:
                ax.set_xlabel(column_names[j], fontsize=19, weight='bold')
                ax.xaxis.set_label_position('top')

            if j > 0:
                ax.set_title(f'NC = {nc(images[i][0], images[i][j]):.3f}',
                             y=-0.15, fontsize=17)

            ax.set_xticks([])
            ax.set_yticks([])

    if row_names is not None:
        for i in range(len(row_names)):
            axes[i][0].set_ylabel(row_names[i], fontsize=19, weight='bold')

    plt.tight_layout()
    plt.show()


def plot_watermarks(images, rows=None, cols=1, figsize=(16, 16), names=None,
                    title='Watermarks'):
    if rows is None:
        rows = len(images)
    assert len(images) <= rows * cols

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    fig.suptitle(title, fontsize=14, y=1.03)

    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]

    for i in range(len(images)):
        ax = axes[i // cols][i % cols]
        ax.imshow(images[i], cmap='gray')

        if names:
            ax.set_title(names[i], fontsize=14)

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    if len(images) < rows * cols:
        empty_img = np.ones(images[0].shape)
        empty_img[0, 0] = 0
        for i in range(len(images), rows * cols):
            ax = axes[i // cols][i % cols]
            ax.imshow(empty_img, cmap='gray')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()