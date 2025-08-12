import numpy as np

def simulate_one(seed, T, a_target):
    taus = np.zeros(T)
    tauhats = taus.copy()
    variances = taus.copy()
    covariances = taus.copy()
    generator = np.random.default_rng(seed)
    simulate_y0 = lambda: generator.gamma(2)
    simulate_y1 = lambda: generator.gamma(3)
    simulate_a = lambda p_t: generator.random() < p_t
    calculate_pa = lambda a, p: (p**a) * ((1.-p) ** (not a))
    y0_tm1 = simulate_y0()
    y1_tm1 = simulate_y1()
    p1 = 0.5
    p2 = p1
    p2_unequal = 0.1
    p_tm1 = p1
    a_tm1 = simulate_a(p_tm1)
    pa_tm1 = calculate_pa(a_tm1, p_tm1)
    def get_p1g1_t(a_counter, t):
        # If next a_t=1 will hit target and get placed in unequal group
        if ((a_counter + 1) == a_target) and (t % 2 == 1):
            p1g1_t = p2_unequal
        else:
            p1g1_t = p2            
        return p1g1_t
    a_counter = 0
    p1g1_t = get_p1g1_t(a_counter, -1)
    p0g1_t = 1. - p1g1_t
    p1g0_t = p1  # Always use p1 if a_t = 0
    p0g0_t = 1. - p1g0_t
    p_t = p1g1_t if a_tm1 else p1g0_t
    a_counter += a_tm1
    
    for t in range(T):
        # Simulate outcomes at next time point
        y00_t = simulate_y0()
        y01_t = simulate_y0()
        y10_t = simulate_y1()
        y11_t = simulate_y1()
        tau_t = (y11_t - y01_t + y10_t - y00_t) / 2.
        taus[t] = tau_t
    
        # Simulate treatment
        a_t = simulate_a(p_t)
        pa_t = calculate_pa(a_t, p_t)
    
        # Select outcomes based on treatment assignment
        if a_tm1:
            if a_t:
                y_t = y11_t
            else:
                y_t = y10_t
        else:
            if a_t:
                y_t = y01_t
            else:
                y_t = y00_t
    
        # Form tauhat
        tauhats[t] = (-1)**(1.-a_tm1) * y_t / (2. * pa_tm1 * pa_t)
    
        # Compute variance and covariance estimates
        var1 = p_tm1 * (1.-p_tm1) * ((y11_t+y10_t)/p_tm1 + (y01_t+y00_t)/(1-p_tm1))**2
        var2 = ((y11_t/p1g1_t - y10_t/p0g1_t)**2 * p1g1_t * p0g1_t / p_tm1 +
                (y01_t/p1g0_t - y00_t/p0g0_t)**2 * p1g0_t * p0g0_t / (1.-p_tm1))
        variances[t] = (var1 + var2) / 4.
    
        if t > 0:
            cov1 = ((-1)**(1.-a_tm2)) / pa_tm2 *(
                y1_tm1 / p_tm1 * (y10_t + y11_t) -
                y0_tm1 / (1 - p_tm1) * (y00_t + y01_t))
            cov2 = ((-1)**(1.-a_tm2)) / pa_tm2 * (y0_tm1 + y1_tm1)
            covariances[t] = cov1/4. - tau_t * cov2/2.
        else:
            covariances[t] = 0.
        
        # Create hypothetical treatment probabilities
        p1g1_t = get_p1g1_t(a_counter, t)
        p0g1_t = 1. - p1g1_t
        p1g0_t = p1  # Always use p1 if a_t = 0
        p0g0_t = 1. - p1g0_t

        # Create actual treatment probabilities
        a_counter += a_t
        if a_counter == a_target:
            t_target = t
            equal_group = t_target % 2 == 0
            if not equal_group:
                p2 = p2_unequal
            # Otherwise, p2 need not be modified
        p_tm2 = p_tm1
        p_tm1 = p_t
        pa_tm2 = pa_tm1
        pa_tm1 = pa_t
        p_t = p1g1_t if a_t else p1g0_t
        assert (p_t == p2 if a_t else p_t == p1)
        
        # Update other variables for next iteration
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
    psi = np.mean(variances + 2*covariances)
    psi_naive = np.mean(variances)
    u = tauhat - tau
    z = u / np.sqrt(psi/T)
    result = {
        "psi": psi,
        "psi_naive": psi_naive,
        "tauhat": tauhat,
        "tau": tau,
        "equal_group": equal_group,
        "t_target": t_target}
    return result

def simulate_job(args):
    seed, T, a_target = args
    return simulate_one(seed, T, a_target)