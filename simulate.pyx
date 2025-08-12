# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport pow, sqrt


cdef class BatchedGamma:
    cdef double shape, scale
    cdef int batch_size, _pos
    cdef object _data        # cannot use typed ndarray directly as attribute
    cdef object rng          # np.random.Generator

    def __init__(self, double shape, double scale=1.0, int batch_size=100_000, seed=None):
        self.shape = shape
        self.scale = scale
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self._pos = 0
        self._data = self.rng.gamma(self.shape, self.scale, self.batch_size)

    cpdef double sample(self):
        cdef double val
        if self._pos >= self.batch_size:
            # refill the buffer
            self._data = self.rng.gamma(self.shape, self.scale, self.batch_size)
            self._pos = 0
        # typed local for fast indexing
        cdef np.ndarray[np.double_t, ndim=1] buf = self._data
        val = buf[self._pos]
        self._pos += 1
        return val


cdef class BatchedUniform:
    cdef int batch_size, _pos
    cdef object _data
    cdef object rng

    def __init__(self, int batch_size=100_000, seed=None):
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self._pos = 0
        self._data = self.rng.random(self.batch_size)

    cpdef double sample(self):
        cdef double val
        if self._pos >= self.batch_size:
            # refill the buffer
            self._data = self.rng.random(self.batch_size)
            self._pos = 0
        # typed local for fast indexing
        cdef np.ndarray[np.double_t, ndim=1] buf = self._data
        val = buf[self._pos]
        self._pos += 1
        return val


cdef inline double simulate_y0(BatchedGamma gamma_gen):
    return gamma_gen.sample()

cdef inline double simulate_y1(BatchedGamma gamma_gen):
    return gamma_gen.sample()

cdef inline int simulate_a(BatchedUniform unif_gen, double p_t):
    return unif_gen.sample() < p_t

cdef inline double calculate_pa(int a, double p):
    return p if a else (1.0 - p)

cdef inline double get_p1g1_t(double p2, double p2_unequal, int a_counter, int a_target, int t):
    if ((a_counter + 1) == a_target) and (t % 2 == 1):
        return p2_unequal
    else:
        return p2


@cython.boundscheck(False)
@cython.wraparound(False)
def simulate_one(int seed, int T, int a_target):
    cdef np.ndarray[np.double_t, ndim=1] taus = np.zeros(T, dtype=np.float64)
    cdef np.ndarray[np.double_t, ndim=1] tauhats = np.zeros(T, dtype=np.float64)
    cdef np.ndarray[np.double_t, ndim=1] variances = np.zeros(T, dtype=np.float64)
    cdef np.ndarray[np.double_t, ndim=1] covariances = np.zeros(T, dtype=np.float64)
    cdef int t, a_tm1, a_t, a_tm2, a_counter = 0
    cdef bint equal_group = False
    cdef int t_target = -1
    cdef double p1 = 0.5
    cdef double p2 = p1
    cdef double p2_unequal = 0.1
    cdef double p_tm1, p_tm2, pa_tm1, pa_tm2, p_t
    cdef double p1g1_t, p0g1_t, p1g0_t, p0g0_t
    cdef double y0_tm1, y1_tm1, y_t
    cdef double y00_t, y01_t, y10_t, y11_t
    cdef double tau_t, pa_t, var1, var2, cov1, cov2
    cdef double tauhat, tau, psi, psi_naive, u, z

    # Initialize batched RNGs
    cdef BatchedGamma gamma2_gen = BatchedGamma(2.0, 1.0, 2*T+2, seed)
    cdef BatchedGamma gamma3_gen = BatchedGamma(3.0, 1.0, 2*T+2, seed + 1)
    cdef BatchedUniform unif_gen = BatchedUniform(T+2, seed + 2)

    # init values
    y0_tm1 = simulate_y0(gamma2_gen)
    y1_tm1 = simulate_y1(gamma3_gen)
    p_tm1 = p1
    a_tm1 = simulate_a(unif_gen, p_tm1)
    pa_tm1 = calculate_pa(a_tm1, p_tm1)

    p1g1_t = p1g1_t = get_p1g1_t(p2, p2_unequal, a_counter, a_target, -1)
    p0g1_t = 1. - p1g1_t
    p1g0_t = p1
    p0g0_t = 1. - p1g0_t
    p_t = p1g1_t if a_tm1 else p1g0_t
    a_counter += a_tm1

    for t in range(T):
        # Simulate outcomes
        y00_t = simulate_y0(gamma2_gen)
        y01_t = simulate_y0(gamma2_gen)
        y10_t = simulate_y1(gamma3_gen)
        y11_t = simulate_y1(gamma3_gen)
        tau_t = (y11_t - y01_t + y10_t - y00_t) / 2.0
        taus[t] = tau_t

        # Treatment
        a_t = simulate_a(unif_gen, p_t)
        pa_t = calculate_pa(a_t, p_t)

        # Select outcomes
        if a_tm1:
            y_t = y11_t if a_t else y10_t
        else:
            y_t = y01_t if a_t else y00_t

        sign_tm1 = -1.0 if (1 - a_tm1) else 1.0
        tauhats[t] = sign_tm1 * y_t / (2.0 * pa_tm1 * pa_t)

        # Variance & covariance
        var1 = p_tm1 * (1. - p_tm1) * ((y11_t + y10_t)/p_tm1 + (y01_t + y00_t)/(1. - p_tm1))**2
        var2 = ((y11_t/p1g1_t - y10_t/p0g1_t)**2 * p1g1_t * p0g1_t / p_tm1 +
                (y01_t/p1g0_t - y00_t/p0g0_t)**2 * p1g0_t * p0g0_t / (1. - p_tm1))
        variances[t] = (var1 + var2) / 4.0

        if t > 0:
            sign_tm2 = -1.0 if (1 - a_tm2) else 1.0
            cov1 = sign_tm2 / pa_tm2 * (
                y1_tm1 / p_tm1 * (y10_t + y11_t) -
                y0_tm1 / (1. - p_tm1) * (y00_t + y01_t)
            )
            cov2 = sign_tm2 / pa_tm2 * (y0_tm1 + y1_tm1)
            covariances[t] = cov1/4.0 - tau_t * cov2/2.0
        else:
            covariances[t] = 0.0

        # Update probabilities
        p1g1_t = p1g1_t = get_p1g1_t(p2, p2_unequal, a_counter, a_target, t)
        p0g1_t = 1. - p1g1_t
        p1g0_t = p1
        p0g0_t = 1. - p1g0_t
        a_counter += a_t
        if (a_counter == a_target) and a_t:
            t_target = t
            equal_group = (t_target % 2 == 0)
            if not equal_group:
                p2 = p2_unequal

        # Roll forward variables
        p_tm2 = p_tm1
        p_tm1 = p_t
        pa_tm2 = pa_tm1
        pa_tm1 = pa_t
        p_t = p1g1_t if a_t else p1g0_t
        assert (p_t == p2 if a_t else p_t == p1)
        if a_tm1:
            y0_tm1 = y10_t
            y1_tm1 = y11_t
        else:
            y0_tm1 = y00_t
            y1_tm1 = y01_t
        a_tm2 = a_tm1
        a_tm1 = a_t

    tauhat = np.mean(tauhats)
    tau = np.mean(taus)
    psi = np.mean(variances + 2.0 * covariances)
    psi_naive = np.mean(variances)
    u = tauhat - tau
    z = u / sqrt(psi / T)
    return {
        "psi": psi,
        "psi_naive": psi_naive,
        "tauhat": tauhat,
        "tau": tau,
        "equal_group": equal_group,
        "t_target": t_target
    }
