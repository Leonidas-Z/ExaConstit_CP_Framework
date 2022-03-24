import numpy as np
import importlib
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from pymoo.docs import parse_doc_string
from pymoo.core.plot import Plot
from pymoo.util.misc import set_if_none


class Scatter:

    def __init__(self,
                 angle=(45, 45),
                 figsize=(8, 6),
                 tight_layout=False,
                 bounds=None,
                 cmap="tab10"
                 ):
        
        self.angle = angle

        # change the font of plots to serif (looks better)
        plt.rc('font', family='serif')

        # the matplotlib classes
        self.figsize = figsize

        # the data to plot
        self.to_plot = []

        # whether to plot a legend or apply tight layout
        self.tight_layout = tight_layout

        # the colormap or the color lists to use
        if isinstance(cmap, str):
            self.cmap = matplotlib.cm.get_cmap(cmap)
        else:
            self.cmap = cmap
        if isinstance(self.cmap, ListedColormap):
            self.colors = self.cmap.colors

        # the dimensional of the data
        self.n_dim = None

        # the boundaries for normalization
        self.bounds = bounds


    def _do(self):

        is_1d = (self.n_dim == 1)
        is_2d = (self.n_dim == 2)
        is_3d = (self.n_dim == 3)
        more_than_3d = (self.n_dim > 3)


        # create the figure and axis objects
        if is_1d or is_2d:
            self._init_figure()
        elif is_3d:
            self._init_figure(plot_3D=True)
        elif more_than_3d:
            self._init_figure(n_rows=self.n_dim, n_cols=self.n_dim)


        # now plot data points for each entry
        for k, F in enumerate(self.to_plot):

            if is_1d:
                F = np.column_stack([F, np.zeros(len(F))])
                labels = self.get_labels() + [""]

                self.plot(self.ax, _type, F)
                self.set_labels(self.ax, labels, False)

            elif is_2d:
                self.plot(self.ax, _type, F)
                self.set_labels(self.ax, self.get_labels(), False)

            elif is_3d:
                set_if_none(_kwargs, "alpha", 1.0)

                self.plot(self.ax, _type, F, **_kwargs)
                self.ax.xaxis.pane.fill = False
                self.ax.yaxis.pane.fill = False
                self.ax.zaxis.pane.fill = False

                self.set_labels(self.ax, self.get_labels(), True)

                if self.angle is not None:
                    self.ax.view_init(*self.angle)

            else:
                labels = self.get_labels()

                for i in range(self.n_dim):
                    for j in range(self.n_dim):

                        ax = self.ax[i, j]

                        if i != j:
                            self.plot(ax, _type, F[:, [i, j]], **_kwargs)
                            self.set_labels(ax, [labels[i], labels[j]], is_3d)
                        else:
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.scatter(0, 0, s=1, color="white")
                            ax.text(0, 0, labels[i], ha='center', va='center', fontsize=20)

        return self
    
    
    
    def _get_labels(self):
        if isinstance(self.axis_labels, list):
            if len(self.axis_labels) != self.n_dim:
                raise Exception("Number of axes labels not equal to the number of axes.")
            else:
                return self.axis_labels
        else:
            return [f"${self.axis_labels}_{{{i}}}$" for i in range(1, self.n_dim + 1)]


    def _init_figure(self, n_rows=1, n_cols=1, plot_3D=False, force_axes_as_matrix=False):
        if not plot_3D:
            self.fig, self.ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=self.figsize)
        else:
            importlib.import_module("mpl_toolkits.mplot3d")
            self.fig = plt.figure(figsize=self.figsize)
            self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')

        # if there is more than one figure we represent it as a 2D numpy array
        if (n_rows > 1 or n_cols > 1) or force_axes_as_matrix:
            self.ax = np.array(self.ax).reshape(n_rows, n_cols)

    def plot(self, ax, _type, F, **kwargs):

        is_3d = F.shape[1] == 3
        if _type is None:
            _type = "scatter"

        if _type == "scatter":
            if is_3d:
                ax.scatter(F[:, 0], F[:, 1], F[:, 2], **kwargs)
            else:
                ax.scatter(F[:, 0], F[:, 1], **kwargs)
        else:
            if is_3d:
                ax.plot_trisurf(F[:, 0], F[:, 1], F[:, 2], **kwargs)
            else:
                ax.plot(F[:, 0], F[:, 1], **kwargs)

    def set_labels(self, ax, labels, is_3d):

        # set the labels for each axis
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

        if is_3d:
            ax.set_zlabel(labels[2])


parse_doc_string(Scatter.__init__)
