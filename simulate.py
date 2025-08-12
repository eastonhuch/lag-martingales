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
    p_tm1 = 0.5
    a_tm1 = simulate_a(p_tm1)
    pa_tm1 = calculate_pa(a_tm1, p_tm1)
    p_t = p_tm1
    a_counter = int(a_tm1)
    
    for t in range(T):
        # Simulate outcomes at next time point
        y00_t = simulate_y0()
        y01_t = simulate_y0()
        y10_t = simulate_y1()
        y11_t = simulate_y1()
        tau_t = (
            y11_t - y01_t + y10_t - y00_t) / 2.
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
        tauhats[t] = (-1)**(not a_tm1) * y_t / (2. * pa_tm1 * pa_t)
    
        # Compute variance and covariance estimates
        var1 = (
            (y11_t**2) / (p_tm1*p_t) +
            (y10_t**2) / (p_tm1*(1.-p_t)) +
            (y01_t**2) / ((1.-p_tm1)*p_t) +
            (y00_t**2) / ((1.-p_tm1)*(1.-p_t))
        )
        var2 = (
            (y10_t - y00_t)**2 +  ## 00
            (y10_t - y00_t) * (y11_t - y01_t) * 2 + ##  01
            (y11_t - y01_t)**2 ##  11
        )
        variances[t] = (var1 - var2) / 4.
    
        if t > 0:
            cov1 = ((-1)**(1-a_tm2)) / pa_tm2 *(
                y1_tm1 / p_tm1 * (y10_t + y11_t) -
                y0_tm1 / (1 - p_tm1) * (y00_t + y01_t))
            cov2 = ((-1)**(not a_tm2)) / pa_tm2 * (y0_tm1 + y1_tm1)
            covariances[t] = cov1/4. - tau_t * cov2/2.
        else:
            covariances[t] = 0.
        
        # Update treatment probability
        p_tm2 = p_tm1
        p_tm1 = p_t
        pa_tm2 = pa_tm1
        pa_tm1 = pa_t
        a_counter += a_t
        if a_counter == a_target:
            t_target = t
            equal_group = t_target % 2 == 0
            if equal_group:
                p_t = 0.5
            else:
                p_t = 0.9
    
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
    u = tauhat - tau
    z = u / np.sqrt(psi/T)
    return z, u, psi, tauhat, tau, equal_group, t_target


def simulate_job(args):
    seed, T, a_target = args
    return simulate_one(seed, T, a_target)