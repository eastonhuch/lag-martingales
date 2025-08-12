from simulate import simulate_one

def simulate_job(args):
    seed, T, a_target = args
    return simulate_one(seed, T, a_target)