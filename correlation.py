#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 18:26:22 2020

@author: sebastian
"""
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


def correlation_color_bar():
    fig, ax = plt.subplots(figsize=(11, 0.8))
    fig.subplots_adjust(bottom=0)
    
    cmap = mpl.cm.coolwarm
    norm = mpl.colors.Normalize(vmin=-100, vmax=100)
    
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    return fig


def correlation_plot(df, dot_size=6000):
    
    sns.set_style('white', {'font.family':'monospace'})
    # Add this before your call to map_diag
    __next_colname = iter(df.columns.tolist()).__next__

    def hide_current_axis(*args, **kwds):
        plt.gca().set_visible(False)

    def corrdot(*args, **kwargs):
        corr_r = args[0].corr(args[1], 'pearson')
        corr_text = f"{corr_r:2.2f}".replace("0.", "")
        ax = plt.gca()

        marker_size = abs(corr_r) * dot_size
        ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.9,
                   cmap="coolwarm",
                   # Colormap normalization.
                   vmin=-1, vmax=1,
                   #transform=ax.transAxes
                  )
        font_size = abs(corr_r) * 15 + 5
        ax.annotate(corr_text,
                    xy=(.50, .5,),
                    #xycoords=ax.transAxes,
                    #xycoords="axes fraction",
                    ha='center',
                    va='center',
                    fontsize=font_size)


    def annotate_colname(x, **kws):
        ax = plt.gca()
        ax.set_axis_off()
        ax.annotate(__next_colname(),
                    xy=(.5, .5),
                    xycoords=ax.transAxes,
                    fontweight='bold',
                    fontsize=16,
                    ha='center',
                    va='center')

    def cor_matrix(df):
        g = sns.PairGrid(df, palette=['red'], height=1.5, aspect=1, despine=False)
        g.map_diag(annotate_colname)
        g.map_lower(corrdot)

        g.map_upper(hide_current_axis)
        # Remove axis labels, as they're in the diagonals.
        for ax in g.axes.flatten():
            ax.set_ylabel('')
            ax.set_xlabel('')

        for ax in g.axes.flatten():
            ax.set_yticklabels([])
            ax.set_xticklabels([])


        # Less space between grid elements.
        g.fig.subplots_adjust(wspace=0.05, hspace=0.05)
        return g
    
    g = cor_matrix(df)
    return g