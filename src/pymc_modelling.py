import numpy as np
import pymc as pm

def get_VAR_arr(data: np.array, n_lags: int) -> np.array:
    return np.concatenate([data[i:len(data) - (n_lags - i)] for i in range(1, n_lags + 1)], axis=1)


def get_samp(max_dim, size=100):
    return np.random.randint(0, max_dim, min(size, max_dim))

def get_gp_smoothing(y: np.array):
    X = np.linspace(0, 2, len(y))[:, None]

    with pm.Model() as gp_mod:
        ell = pm.Gamma("ell", alpha=2, beta=1)
        eta = pm.HalfNormal("eta", sigma=5)

        cov = eta**2 * pm.gp.cov.ExpQuad(1, ell)
        gp = pm.gp.Latent(cov_func=cov)

        f = gp.prior("f", X=X)

        sigma = pm.HalfNormal("sigma", sigma=2.0)
        nu = 1 + pm.Gamma(
            "nu", alpha=2, beta=0.1
        )  # add one because student t is undefined for degrees of freedom less than one
        obs = pm.Deterministic('obs', nu)
        y_ = pm.StudentT("y", mu=f, lam=1.0 / sigma, nu=nu, observed=y)

        prior = pm.sample_prior_predictive()
        trace = pm.sample(1000, nuts_sampler="numpyro", tune=1000, chains=2)
        post = pm.sample_posterior_predictive(trace)
        
    return gp_mod, prior, trace, post