import os
import pickle
import matplotlib
import matplotlib.pyplot as plt

import torch
import pyro
from pyro.infer import MCMC, NUTS
import numpy as np
import arviz as az

pyro.set_rng_seed(0)

KB = 0.0019872041        # in kcal/mol/K
INJ_VOL = 1.2e-5         # in liter
CELL_CONCENTR = 0.1      # milli molar
SYRINGE_CONCENTR = 1.0   # milli molar

HEAT_FILE = "~/inputs/Mg1EDTAp1a.DAT"
OUT_DIR = "~/outputs"

def load_heat_micro_cal(origin_heat_file):
    """
    :param origin_heat_file: str, name of heat file
    :return: tensor array, heats in micro calorie
    """

    heats = []
    with open(origin_heat_file) as handle:
        handle.readline()
        for line in handle:
            if len(line.split()) == 6:
                heats.append(np.float(line.split()[0]))

    return torch.as_tensor(heats)

q_actual_micro_cal = load_heat_micro_cal(HEAT_FILE)
q_actual_cal = q_actual_micro_cal * 1e-6
q_actual_cal

n_injections = len(q_actual_cal)
print("# injections:", n_injections)
injection_volumes = [INJ_VOL for _ in range(n_injections)]

"""## Functions"""

def heats_TwoComponentBindingModel(V0, DeltaVn, P0, Ls, DeltaG, DeltaH, DeltaH_0, beta, N):
    """
    Expected heats of injection for two-component binding model.

    ARGUMENTS
    V0 - cell volume (liter)
    DeltaVn - injection volumes (liter)
    P0 - Cell concentration (millimolar)
    Ls - Syringe concentration (millimolar)
    DeltaG - free energy of binding (kcal/mol)
    DeltaH - enthalpy of binding (kcal/mol)
    DeltaH_0 - heat of injection (cal)
    beta - inverse temperature * gas constant (mole / kcal)
    N - number of injections

    Returns
    -------
    expected injection heats (calorie)

    """
    Kd = torch.exp(beta * DeltaG)   # dissociation constant (M)

    # Compute complex concentrations.
    # Pn[n] is the protein concentration in sample cell after n injections
    # (M)
    Pn = torch.zeros([N])
    # Ln[n] is the ligand concentration in sample cell after n injections
    # (M)
    Ln = torch.zeros([N])
    # PLn[n] is the complex concentration in sample cell after n injections
    # (M)
    PLn = torch.zeros([N])

    dcum = 1.0  # cumulative dilution factor (dimensionless)
    for n in range(N):
        # Instantaneous injection model (perfusion)
        # dilution factor for this injection (dimensionless)
        d = 1.0 - (DeltaVn[n] / V0)
        dcum *= d  # cumulative dilution factor
        # total quantity of protein in sample cell after n injections (mol)
        P = V0 * P0 * 1.e-3 * dcum
        # total quantity of ligand in sample cell after n injections (mol)
        L = V0 * Ls * 1.e-3 * (1. - dcum)
        
        # complex concentration (M)
        PLn[n] = (0.5 / V0 * ((P + L + Kd * V0) - torch.sqrt((P + L + Kd * V0) ** 2 - 4 * P * L) ))

        # free protein concentration in sample cell after n injections (M)
        Pn[n] = P / V0 - PLn[n]

        # free ligand concentration in sample cell after n injections (M)
        Ln[n] = L / V0 - PLn[n]

    # Compute expected injection heats.
    # q_n_model[n] is the expected heat from injection n
    q_n = torch.zeros([N])
    # Instantaneous injection model (perfusion)
    # first injection
    q_n[0] = (DeltaH * V0 * PLn[0])*1000. + DeltaH_0

    for n in range(1, N):
        d = 1.0 - (DeltaVn[n] / V0)  # dilution factor (dimensionless)
        # subsequent injections
        q_n[n] = (DeltaH * V0 * (PLn[n] - d * PLn[n - 1])) * 1000. + DeltaH_0

    return q_n

def logsigma_guesses(q_n_cal):
    log_sigma_guess = np.log(q_n_cal[-4:].std())
    log_sigma_min = log_sigma_guess - 10
    log_sigma_max = log_sigma_guess + 5
    return log_sigma_min, log_sigma_max

def deltaH0_guesses(q_n_cal):
    heat_interval = (q_n_cal.max() - q_n_cal.min())
    DeltaH_0_min = q_n_cal.min() - heat_interval
    DeltaH_0_max = q_n_cal.max() + heat_interval
    return DeltaH_0_min, DeltaH_0_max

def lognormal_prior(name, stated_value, uncertainty):
    """
    :param name: str
    :param stated_value: float
    :uncertainty: float
    :rerurn: pyro.Lognormal
    """
    m = torch.as_tensor(stated_value)
    v = torch.as_tensor(uncertainty ** 2) 
    mu = torch.log(m / torch.sqrt(1 + (v / (m ** 2))))
    tau = torch.sqrt(torch.log(1 + (v / (m ** 2))))
    return pyro.sample(name, pyro.distributions.LogNormal(loc=mu, scale=tau))

def uniform_prior(name, lower, upper):
    """
    :param name: str
    :param lower: float
    :param upper: float
    :return: pyro.Uniform
    """
    return pyro.sample(name, pyro.distributions.Uniform(low=lower, high=upper))

def param(injection_volumes, cell_concentration, syringe_concentration,
          cell_volume, temperature, dcell, dsyringe,
          uniform_P0=False, P0_min=None, P0_max=None, 
          uniform_Ls=False, Ls_min=None, Ls_max=None):
    """
    :param cell_concentration: concentration of the sample cell in milli molar, float
    :param syringe_concentration: concentration of the syringe in milli molar, float
    :param cell_volume: volume of sample cell in liter, float #check the instrutment 
    :param temperature: temprature in kelvin, float
    :param dcell: relative uncertainty in cell concentration, float
    :param dsyringe: relative uncertainty in syringe concentration, float
    :param uniform_P0: if True, use uniform prior for cell concentration, bool
    :param P0_min: only use if uniform_P0 is True, float
    :param P0_max: only use if uniform_P0 is True, float
    :param uniform_Ls: if True, use uniform prior for syringe concentration, bool
    :param Ls_min: only use if uniform_Ls is True, float
    :param Ls_max: only use if uniform_Ls is True, float

    :return: priors for P0, Ls, DeltaG, DeltaH, DeltaH_0, log_sigma
    """
    if uniform_P0 and (P0_min is None or P0_max is None):
        raise ValueError("If uniform_P0 is True, both P0_min and P0_max must be provided")
    
    if uniform_Ls and (Ls_min is None or Ls_max is None):
        raise ValueError("If uniform_Ls is True, both Ls_min and Ls_max must be provided")

    DeltaH_0_min, DeltaH_0_max = deltaH0_guesses(q_actual_cal)
    log_sigma_min, log_sigma_max = logsigma_guesses(q_actual_cal)

    stated_P0 = cell_concentration
    uncertainty_P0 = dcell * stated_P0

    stated_Ls = syringe_concentration
    uncertainty_Ls = dsyringe * stated_Ls
    
    # prior for receptor concentration
    if uniform_P0:
        P0 = uniform_prior("P0", lower=P0_min, upper=P0_max)
    else:
        P0 = lognormal_prior("P0", stated_value=stated_P0, uncertainty=uncertainty_P0)

    # prior for ligand concentration
    if uniform_Ls:
        Ls = uniform_prior("Ls", lower=Ls_min, upper=Ls_max)
    else:
        Ls = lognormal_prior("Ls", stated_value=stated_Ls, uncertainty=uncertainty_Ls)
  
    # priors for DeltaG, DeltaH, DeltaH_0, log_sigma
    DeltaG = uniform_prior("DeltaG", lower=-40., upper=4.)
    DeltaH = uniform_prior("DeltaH", lower=-100., upper=100.)
    DeltaH_0 = uniform_prior("DeltaH_0", lower=DeltaH_0_min, upper=DeltaH_0_max)
    log_sigma = uniform_prior("log_sigma", lower=log_sigma_min, upper=log_sigma_max)

    return P0, Ls, DeltaG, DeltaH, DeltaH_0, log_sigma

def make_TwoComponentBindingModel(q_actual_cal, 
                                  injection_volumes, 
                                  cell_concentration, 
                                  syringe_concentration,
                                  cell_volume=0.001434,
                                  temperature=298.15,
                                  dcell=0.1, dsyringe=0.1,
                                  uniform_P0=False, P0_min=None, P0_max=None, 
                                  uniform_Ls=False, Ls_min=None, Ls_max=None):
  
    """
    to create a model
    :param q_actual_cal: observed heats in calorie, array-like
    :param injection_volumes: injection volumes in liter, array-like
    :param cell_concentration: concentration of the sample cell in milli molar, float
    :param syringe_concentration: concentration of the syringe in milli molar, float
    :param cell_volume: volume of sample cell in liter, float #check the instrutment 
    :param temperature: temprature in kelvin, float
    :param dcell: relative uncertainty in cell concentration, float
    :param dsyringe: relative uncertainty in syringe concentration, float
    :param uniform_P0: if True, use uniform prior for cell concentration, bool
    :param P0_min: only use if uniform_P0 is True, float
    :param P0_max: only use if uniform_P0 is True, float
    :param uniform_Ls: if True, use uniform prior for syringe concentration, bool
    :param Ls_min: only use if uniform_Ls is True, float
    :param Ls_max: only use if uniform_Ls is True, float
    
    :return: an instance of pyro.model
    """
    assert len(q_actual_cal) == len(injection_volumes), "q_actual_cal and injection_volumes must have the same len."
          
    V0 = cell_volume
    DeltaVn = injection_volumes
    beta = 1 / KB / temperature
    n_injections = len(q_actual_cal)
    
    P0, Ls, DeltaG, DeltaH, DeltaH_0, log_sigma = param(injection_volumes, cell_concentration, syringe_concentration, 
                                                        cell_volume, temperature, dcell, dsyringe,
                                                        uniform_P0, P0_min, P0_max,
                                                        uniform_Ls, Ls_min, Ls_max)

    sigma_cal = torch.exp(log_sigma)
    q_model_cal = heats_TwoComponentBindingModel(V0, DeltaVn, P0, Ls, DeltaG, DeltaH, DeltaH_0, beta, n_injections)
    
    pyro.sample('q_obs', pyro.distributions.Normal(loc=q_model_cal, scale=sigma_cal), obs=q_actual_cal)
    return P0, Ls, DeltaG, DeltaH, DeltaH_0, log_sigma

nuts_kernel = NUTS(make_TwoComponentBindingModel)
mcmc = MCMC(nuts_kernel, warmup_steps=10, num_samples=50, num_chains=4)
mcmc.run(q_actual_cal, injection_volumes, CELL_CONCENTR, SYRINGE_CONCENTR)
mcmc.summary()

trace = mcmc.get_samples(group_by_chain=True)
pickle.dump(make_TwoComponentBindingModel, open(os.path.join(OUT_DIR, "pyro.pickle"), "wb"))
pickle.dump(trace, open(os.path.join(OUT_DIR, "pyro_trace.pickle"), "wb"))

data = az.convert_to_inference_data(trace)
az.plot_trace(data)
plt.savefig(os.path.join(OUT_DIR, "pyro_trace.pdf"))