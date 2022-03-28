import numpy as np

from visualization.staff.docs import parse_doc_string
from visualization.staff.plot import Plot, set_if_none


class Scatter(Plot):

    def __init__(self,
                 epsdot=1e-3,
                 custom_dt=None,
                 angle=(45, 45),
                 **kwargs):
        """

        Scatter Plot

        Parameters
        ----------------

        axis_style : {axis_style}
        endpoint_style : dict
            Endpoints are drawn at each extreme point of an objective. This style can be modified.
        labels : {labels}

        Other Parameters
        ----------------

        figsize : {figsize}
        title : {title}
        legend : {legend}
        tight_layout : {tight_layout}
        cmap : {cmap}
        """        
        
        super().__init__(**kwargs)
    
        if custom_dt == None and epsdot == None:
            print("one of the custom_dt and epsdot must not be None")
        
        self.angle = angle
        self.epsdot = epsdot
        self.custom_dt = custom_dt


    def _do(self):

        is_1d = (self.n_dim == 1)
        is_2d = (self.n_dim == 2)
        is_3d = (self.n_dim == 3)
        more_than_3d = (self.n_dim > 3)

        # find strain for each  

        # create the figure and axis objects
        if is_1d or is_2d:
            self.init_figure()
        elif is_3d:
            self.init_figure(plot_3D=True)
        elif more_than_3d:
            self.init_figure(n_rows=self.n_dim, n_cols=self.n_dim)

        # now plot data points for each entry
        for k, (S_sim, kwargs) in enumerate(self.to_plot):

            # copy the arguments and set the default color
            _kwargs = kwargs.copy()
            set_if_none(_kwargs, "color", self.colors[k % len(self.colors)])

            # determine the plotting type - scatter or line
            _type = _kwargs.get("plot_type")
            if "plot_type" in _kwargs:
                del _kwargs["plot_type"]

            if is_1d:
                S_sim = np.column_stack([S_sim, np.zeros(len(S_sim))])
                labels = self.get_labels() + [""]

                self.plot(self.ax, _type, S_sim, **_kwargs)
                self.set_labels(self.ax, labels, False)

            elif is_2d:
                self.plot(self.ax, _type, S_sim, **_kwargs)
                self.set_labels(self.ax, self.get_labels(), False)

            elif is_3d:
                set_if_none(_kwargs, "alpha", 1.0)

                self.plot(self.ax, _type, S_sim, **_kwargs)
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
                            self.plot(ax, _type, S_sim[:, [i, j]], **_kwargs)
                            self.set_labels(ax, [labels[i], labels[j]], is_3d)
                        else:
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.scatter(0, 0, s=1, color="white")
                            ax.text(0, 0, labels[i], ha='center', va='center', fontsize=20)

        return self


    def plot(self, ax, _type, S_sim, **kwargs):
        # kwargs includes all the addtional arguments for the matplotlib class (color, size, etc.)
        is_3d = S_sim.shape[1] == 3
        if _type is None:
            _type = "scatter"

        if _type == "scatter":
                ax.scatter(S_sim[:, 0], S_sim[:, 1], **kwargs)
        else:
                ax.plot(S_sim[:, 0], S_sim[:, 1], **kwargs)

    def set_labels(self, ax, labels, is_3d):

        # set the labels for each axis
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

        if is_3d:
            ax.set_zlabel(labels[2])

    def find_strains(self, S_sim, epsdot, custom_dt):

        if custom_dt!=None:
            time = np.loadtxt(self.custom_dt)

        elif epsdot!=None:
            nsteps = S_sim.shape[0]
            eps = np.zeros(nsteps)
            time = np.ones(nsteps)

            for i in range(1, nsteps):
                dtime = time[i]
                if S_sim[i] - S_sim[i - 1] >= 0:
                    eps[i] = eps[i - 1] + epsdot * dtime
                else:
                    eps[i] = eps[i - 1] - epsdot * dtime
        return eps


        

