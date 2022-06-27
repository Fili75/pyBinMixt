import example as e
import numpy as np
import sklearn.metrics as sk


class BinMixt:
    """
    Bin-marginal mixture model
    Estimate the parameter of a mixture model after a frugal compression in bin-marginal data, build
    from univariate marginal grids.
    It uses a composite likelihood EM (CL-EM) on bin-marginal data.
    It has an initialization phase where a fixed points are randomly extracted and used as initizations of several small CL-EM.
    The best point is retained and used to initialize a second long CL-EM
    Parameters
    ----------
    classes : int
        The number of mixture components.
    grid : list
        List of integers specyfing the refinement of the marginal grids. The length of the list must be equal
        to the dimension of the analyzed data.
    it_init : int
        Number of iterations of the small CL-EM algorithm
    it_algo : int
        Number of iterations of the final CL-EM algorithm
    eps_init : float
        Tolerance of the small CL-EM algorithm
    eps_algo : float
       Tolerance of the final CL-EM algorithm
    n_init : int
       Number of initialization points for the small CL-EM algorithms
    seed : int
       Fix the seed for reproducibility
    Attributes
    ----------
    bin_ : dict
         Bin-marginal data
    pi_ : list
        The weights of each mixture components
    mu_ : list
        The means of each mixture components
    v_ : list
        The variances of each mixture components
    loglik_ : list
        Sequence of bin-marginal composite likelihood given by the CL-EM algorithm
    --------
    """

    def __init__(
            self,
            classes,
            grid,
            it_init,
            it_algo,
            eps_init,
            eps_algo,
            n_init,
            seed=1):
        if isinstance(grid, np.ndarray):
            self.grid = grid.tolist()
        self.classes = classes
        self.grid = grid
        self.it_init = it_init
        self.it_algo = it_algo
        self.eps_init = eps_init
        self.eps_algo = eps_algo
        self.n_init = n_init
        self.seed = seed

    def get_params(self):
        return {
            "classes": self.classes,
            "grid": self.grid,
            "it_init": self.it_init,
            "it_algo": self.it_algo,
            "eps_init": self.eps_init,
            "eps_algo": self.eps_algo,
            "n_init": self.n_init,
            "seed": self.seed}

    def fit(self, X):
        """
        :param X: list
                Input data
        :return: self: object
                The fitted mixture
        """
        if isinstance(X, np.ndarray):
            X = X.tolist()
        result = e.binmixtclassic(
            X,
            self.classes,
            self.grid,
            self.it_init,
            self.eps_init,
            self.eps_algo,
            self.it_algo,
            self.n_init,
            self.seed)
        self.bin_ = result["bin"]
        self.pi_ = result["pi"]
        self.mu_ = result["mu"]
        self.v_ = result["v"]
        self.loglik_ = result["loglik"]

    def fit_predict(self, X):
        """
        :param X: list
                Input data
        :return: labels: list
                estimated labels
        """
        if isinstance(X, np.ndarray):
            X = X.tolist()
        result = e.binmixtclassic(
            X,
            self.classes,
            self.grid,
            self.it_init,
            self.eps_init,
            self.eps_algo,
            self.it_algo,
            self.n_init,
            self.seed)
        self.bin_ = result["bin"]
        self.pi_ = result["pi"]
        self.mu_ = result["mu"]
        self.v_ = result["v"]
        self.loglik_ = result["loglik"]
        prob = e.posterior(X, self.pi_, self.mu_, self.v_)
        return np.argmax(prob, axis=1)

    def predict(self, X):
        """
        :param X: list
                Input data
        :return: labels: list
                estimated labels
        """
        if isinstance(X, np.ndarray):
            X = X.tolist()
        prob = e.posterior(X, self.pi_, self.mu_, self.v_)
        return np.argmax(prob, axis=1)

    def score(self, X, y):
        """
        :param X: list
                Input data
        :param  y: list
                Input labels
        :return: labels: list
                estimated labels
        """
        if isinstance(X, np.ndarray):
            X = X.tolist()
        prob = e.posterior(X, self.pi_, self.mu_, self.v_)
        pred = np.argmax(prob, axis=1)
        return sk.adjusted_rand_score(y, pred)


class WindowTransformer:
    """
    Tranform the given dataset in a new dataset containing variances, means and root mean squared of points inside
    sliding windows of fixed width for each variable.
    --------
    Parameters
    width : int
        The width of the sliding window
    include_sd : bool
        Standard deviation is computed. Default is True
    include_m : bool
        Mean is computed. Default is True
    include_rms : bool
        Root Mean Square is computed. Default is True
    """

    def __init__(
            self,
            width,
            include_sd=True,
            include_m=True,
            include_rms=True):
        self.width = width
        self.include_sd = include_sd
        self.include_m = include_m
        self.include_rms = include_rms

    def get_params(self):
        return {
            "width": self.width,
            "include_sd": self.include_sd,
            "include_m": self.include_m,
            "include_rms": self.include_rms}

    def fit(self, X, y):
        """
        :param X: list
                Input data
        :param  y: list
                Input labels
        :return: self: object
                The estimated transformation
        """
        if isinstance(X, np.ndarray):
            X = X.tolist()
        if isinstance(y, np.ndarray):
            y = y.tolist()
        self.trans = e.window(
            X,
            y,
            self.width,
            self.include_sd,
            self.include_m,
            self.include_rms)

    def fit_transform(self, X, y):
        """
        :param X: list
                Input data
        :param  y: list
                Input labels
        :return: trans: list
                the trasformed dataset
        """
        if isinstance(X, np.ndarray):
            X = X.tolist()
        if isinstance(y, np.ndarray):
            y = y.tolist()
        self.trans = e.window(
            X,
            y,
            self.width,
            self.include_sd,
            self.include_m,
            self.include_rms)
        return self.trans
