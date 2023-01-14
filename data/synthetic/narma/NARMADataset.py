import numpy as np
import matplotlib.pyplot as plt
import os


class NARMA():
    """
    Non-linear Autoregressive Moving Average generator.

    An n-th order signal generator of the NARMA class, as defined in [1]_:

    .. \math:: y(k+1) = a[0] * y(k) + a[1] * y(k) * \sum_{i=0}^{n-1} y(k-i) +
    a[2] * u(k-(n-1)) * u(k) + a[3]

    where u is generated from Uniform(0, 1). NOTE: Only supports regular time
    samples.  

    Parameters
    ----------
    order : int (default 10) The order (n) of non-linear interactions as
        described in the formula above. coefficients : iterable (default [0.3,
        0.05, 1.5, 0.1]) The coefficients denoted by iterable `a` in the formula
        above. As in [1]_. initial_condition : iterable or None (default None)
        An array of starting values of y(k-n) until y(k). The default is an aray
        of zeros. seed : int Use this seed to recreate any of the internal
        errors.

    Attributes
    ----------
    errors : numpy array or None Random number sequence that was used to
        generate last NARMA sequence.

    References
    ----------
    .. [1]
    http://ieeexplore.ieee.org.ezp-prod1.hul.harvard.edu/stamp/stamp.jsp?arnumber=846741

    Acknowledgement
    ----------
    This code file is from
    https://github.com/TimeSynth/TimeSynth/blob/master/timesynth/signals/narma.py,
    with only a little modification.
    """

    # Constructor
    def __init__(self, sample_len, system_order=10, coefficients=[0.3, 0.05, 1.5, 0.1], seed=0):

        # Properties
        self.sample_len = sample_len
        self.system_order = system_order
        self.coefficients = np.array(coefficients)
        self.order = system_order
        self.random = np.random.RandomState(seed)

        self.noise_condition = np.random.uniform(0, 1, size=self.order)
        self.initial_condition = np.zeros(self.order)

        self.sample = self.sample_vectorized(np.zeros(sample_len))

    # end __init__

    # endregion CONSTRUCTORS

    def _next_value(self, values, rands, index):
        """Internal short-hand method to calculate next value."""
        # Short-hand parameters
        n = self.order
        a = self.coefficients

        # Get value arrays
        i = index
        y = values
        u = rands

        # Compute next value
        return a[0] * y[i-1] + a[1] * y[i-1] * np.sum(y[i-n:n]) + a[2] * u[i-n] * u[i] + a[3]

    def sample_vectorized(self, times):
        """Samples for all time points in input
        
        Internalizes Uniform(0, 0.5) random distortion for u.
        Parameters
        ----------
        times: array like
            all time stamps to be sampled
        
        Returns
        -------
        samples : numpy array
            samples for times provided in time_vector
        """
        # Set bounds
        start = self.initial_condition.shape[0]

        # Get relevant arrays
        inits = self.initial_condition
        rand_inits = self.noise_condition
        rands = np.concatenate(
            (rand_inits, self.random.uniform(0, 1, size=times.shape[0])))
        values = np.concatenate((inits, np.zeros(times.shape[0])))

        # Sample step-wise
        end = values.shape[0]
        for t in range(start, end):
            values[t] = self._next_value(values, rands, t)

        # Store valus for later retrieval
        # self.errors = rands[start:]

        # Return trimmed values (exclude initial condition)
        samples = values[start:]
        return samples

    # region PRIVATE

    def show(self):
        plt.plot(self.sample)
        plt.show()

    def save(self, location):
        # 将p_all_best写入excel
        # data1 = {}
        # data1['time series'] = list(self.sample)
        # df1 = pd.DataFrame(data1)
        # # 写入数据
        # with pd.ExcelWriter(location) as writer:
        #     df1.to_excel(writer, sheet_name='NARMADataset')
        # writer.save()
        # writer.close()

        np.save(os.path.join('data/paper.esn', location), self.sample)


if __name__ == "__main__":
    dataset = NARMA(5600)
    # location = "NARMADataset.xlsx"
    # dataset.save(location)
    # dataset.show()
    dataset.show()
    # dataset.save('narma')
