import numpy as np

from Visualization.staff.docs import parse_doc_string
from Visualization.staff.plot import Plot, set_if_none


class Stress(Plot):

    def __init__(self,
                 epsdot=1e-3,
                 custom_dt=None,
                 angle=(45, 45),
                 **kwargs):
        



        
        super().__init__(**kwargs)
    
        if custom_dt == None and epsdot == None:
            raise Exception("One of the custom_dt and epsdot must not be None")
        
        self.angle = angle
        self.epsdot = epsdot
        self.custom_dt = custom_dt


    def _do(self):

        # find strain for each  

        # create the figure and axis objects
        self.init_figure()

        # now plot data points for each entry
        for k, (stress, kwargs) in enumerate(self.to_plot):
            
            # copy the arguments and set the default color
            _kwargs = kwargs.copy()
            set_if_none(_kwargs, "color", self.colors[k % len(self.colors)])

            # determine the plotting type - scatter or line
            _type = _kwargs.get("plot_type")
            if "plot_type" in _kwargs:
                del _kwargs["plot_type"]
            
            # determine strains
            self.strain = self.find_strains(stress, self.epsdot, self.custom_dt)
            
            # plot
            self.plot(self.ax, _type, stress, **_kwargs)
            self.set_labels(self.ax, self.get_labels())
            print("\n")
            #print(stress)
            print("\n")
            #print(stress)
            #print(self.to_plot[1])
        return self


    def plot(self, ax, _type, stress, **kwargs):
        ax.grid()
        if _type is None:
            _type = "scatter"

        if _type == "scatter":
                ax.scatter(self.strain, stress, **kwargs)
        else:
                ax.plot(self.strain, stress, **kwargs)
        

    def set_labels(self, ax, labels):

        # set the labels for each axis
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])


    def find_strains(self, S_sim, epsdot, custom_dt):
        print('\n')
        S_sim = np.array(S_sim[0])
        print(S_sim)
        if custom_dt!=None:
            time = np.loadtxt(self.custom_dt)
    
        elif epsdot!=None:
            nsteps = len(S_sim)
            eps = np.zeros(nsteps)
            time = np.ones(nsteps)

            for i in range(1, nsteps):
                dtime = time[i]
                if S_sim[i] - S_sim[i - 1] >= 0:
                    eps[i] = eps[i - 1] + epsdot * dtime
                else:
                    eps[i] = eps[i - 1] - epsdot * dtime
        return eps

