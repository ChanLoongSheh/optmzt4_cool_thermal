import numpy as np
from mystic.solvers import DifferentialEvolutionSolver2
from mystic.termination import VTR
from mystic.monitors import Monitor, VerboseMonitor
from mystic.strategy import Rand1Bin
from mystic.tools import list_or_tuple_or_ndarray

class AdaptiveDESolver(DifferentialEvolutionSolver2):
    def __init__(self, ndim, npop, seed=None):
        super(AdaptiveDESolver, self).__init__(ndim, npop)
        self._random = np.random.RandomState(seed)  # Optional fixed seed for reproducibility
        self.last_constraint = 0

    def Solve(self, cost=None, termination=None, ExtraArgs=None, **kwds):
        self.SetObjective(cost)
        while not self.Terminated():
            self.Step(cost, **kwds)
            if self.Terminated():
                print("Termination condition met. Exiting loop.")
                break  # Ensure we exit the loop when termination condition is met

    def Step(self, cost=None, ExtraArgs=(), **kwds):
        """Override to inject variable scale factor."""
        # Use a random scale factor within [0.5, 1.0] for each generation
        self.scale = self._random.uniform(0.5, 1.0)
        # Continue with the standard stepping process
        super(AdaptiveDESolver, self).Step(cost, ExtraArgs=ExtraArgs, **kwds)

class CustomMonitor(Monitor):
    """A verbose version of the basic Monitor.

Prints output 'y' every 'interval', and optionally prints
input parameters 'x' every 'xinterval'.
    """
    def __init__(self, interval=10, xinterval=np.inf, all=True, **kwds):
        super(CustomMonitor, self).__init__(**kwds)
        if not interval or interval is np.nan: interval = np.inf
        if not xinterval or xinterval is np.nan: xinterval = np.inf
        self._yinterval = interval
        self._xinterval = xinterval
        self._all = all
        self.storestep = 0
        return
    def info(self, message):
        super(CustomMonitor,self).info(message)
        print("%s" % "".join(["",str(message)]))
        return
    def __call__(self, x, y, id=None, solver=None, best=0, k=False):
        super(CustomMonitor, self).__call__(x, y, id, k=k) # once call the CustomMonitor, the self._step plus 1
        if self._yinterval is not np.inf and \
           int((self._step-1) % self._yinterval) == 0:
            if not list_or_tuple_or_ndarray(y):
                who = ''
                y = " %s" % self._ik(self._y[-1], k)
            elif self._all:
                who = ''
                y = " %s" % self._ik(self._y[-1], k)
            else:
                who = ' best'
                y = " %s" % self._ik(self._y[-1][best], k)
            msg = "Generation {0} has {1} {2}: {3}, constraint:{4}".format(self._step-1, who, self.label, y, solver)
            if id is not None: msg = "[id: %s] " % (id) + msg
            print(msg)
        if self._xinterval is not np.inf and \
           int((self._step-1) % self._xinterval) == 0:
            if not list_or_tuple_or_ndarray(x):
                who = ''
                x = " %s" % self._x[-1]
            elif self._all:
                who = ''
                x = "\n %s" % self._x[-1]
            else:
                who = ' best'
                x = "\n %s" % self._x[-1][best]
            msg = "Generation %s has%s fit parameters:%s" % (self._step-1,who,x)
            if id is not None: msg = "[id: %s] " % (id) + msg
            print(msg)
        return
    pass

def example_cost_function(x):
    """Example cost function: Rastrigin's function."""
    A = 10
    return A * len(x) + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])

if __name__ == "__main__":
    ndim = 4  # Number of dimensions
    npop = 40  # Population size
    maxiter = 2000  # Maximum number of iterations
    stepmon = VerboseMonitor(10)

    solver = AdaptiveDESolver(ndim, npop)
    solver.SetGenerationMonitor(stepmon)
    solver.SetRandomInitialPoints(min=[-5.12] * ndim, max=[5.12] * ndim)
    termination = VTR(0.01)
    solver.SetTermination(termination)
    solver.Solve(example_cost_function, strategy=Rand1Bin)

    print("Best Solution: {}".format(solver.bestSolution))
    print("Best Objective: {}".format(solver.bestEnergy))