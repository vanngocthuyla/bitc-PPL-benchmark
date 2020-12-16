# bitc

Comparison of the three PPLs (numpyro, pyro and pymc3) for running the ITC (isothermal titration calorimetry) data

## 1. Introduction to the data

Data of ITC to running the Bayesian model: [Mg1EDTAp1a.DAT](https://github.com/vanngocthuyla/bitc/tree/main/inputs/Mg1EDTAp1a.DAT)

## 2. Bayesian Models

[Reference 1](https://github.com/choderalab/bayesian-itc) and [Reference 2](https://github.com/nguyentrunghai/bayesian-itc/tree/d8cbf43240862e85d72d7d0c327ae2c6f750e600) 

## 3. Installation of three PPLs

- Numpyro: 
Numpyro v0.4.1, Numpy v1.18.5, Matplotlib v3.2.2, Arviz v0.10.0
- Pyro:
Pyro v1.5.1, Torch v1.7.0, Numpy v1.18.5, Matplotlib v3.2.2, Arviz v0.10.0
- Pymc3:
Pymc3 v3.8, Theano v1.0.5, Pandas v0.25, Arviz v0.4.1
- Python scripts: [numpyro](https://github.com/vanngocthuyla/bitc/blob/main/scripts/bitc_numpyro.py), [pyro](https://github.com/vanngocthuyla/bitc/blob/main/scripts/bitc_pyro.py), [pymc3](https://github.com/vanngocthuyla/bitc/blob/main/scripts/bitc_pymc3.py),

## 4. Comparison of 3 PPLs

### Checking the convergence of 3 PPLs
- Numpyro

<img src='https://vanngocthuyla.github.io/Data_Analysis/images/bayesian/Numpyro_Plot.png' width="800">

|Parameter|mean|std|median|5.0%|95.0%|n_eff|r_hat|
|---------|----|---|------|----|-----|-----|-----|
|DeltaG|-8.99|0.08|-8.99|-9.12|-8.87|7859.73|1.00|
|DeltaH|-2.10|0.15|-2.10|-2.34|-1.86|5488.17|1.00|
|DeltaH_0|-0.00|0.00|-0.00|-0.00|-0.00|11006.09|1.00|
|Ls|1.12|0.08|1.12|0.98|1.24|5377.76|1.00|
|P0|0.09|0.01|0.09|0.08|0.10|5376.25|1.00|
|log_sigma|-14.78|0.16|-14.79|-15.05|-14.52|8316.83|1.00

- Pyro

<img src='https://vanngocthuyla.github.io/Data_Analysis/images/bayesian/Pyro_Plot.png' width="800">

|Parameter|mean|std|median|5.0%|95.0%|n_eff|r_hat|
|---------|----|---|------|----|-----|-----|-----|
|P0|0.09|0.01|0.09|0.08|0.10|9944.88|1.00|
|Ls|1.12|0.08|1.11|0.99|1.24|9950.80|1.00|
|DeltaG|-8.99|0.08|-8.99|-9.12|-8.87|14202.63|1.00|
|DeltaH|-2.11|0.15|-2.10|-2.35|-1.86|10039.88|1.00|
|DeltaH_0|-0.00|0.00|-0.00|-0.00|-0.00|20708.19|1.00|
|log_sigma|-14.78|0.17|-14.79|-15.05|-14.51|16259.00|1.00|

- Pymc3

<img src='https://vanngocthuyla.github.io/Data_Analysis/images/bayesian/Pymc3_Plot.png' width="800">

|Parameter|mean|sd|hpd_3%|hpd_97%|mcse_mean|mcse_sd|ess_mean|ess_sd|ess_bulk|ess_tail|r_hat|
|---------|----|--|------|-------|---------|-------|--------|------|--------|--------|-----|
|P0|0.088|0.006|0.077|0.100|0.000|0.000|12960.0|12866.0|13083.0|8512.0|1.0|
|Ls|1.119|0.080|0.974|1.271|0.001|0.000|12979.0|12887.0|13100.0|8608.0|1.0|
|DeltaG|-8.992|0.076|-9.135|-8.847|0.001|0.000|21974.0|21973.0|21894.0|34408.0|1.0|
|DeltaH|-2.104|0.151|-2.388|-1.826|0.001|0.001|13464.0|13464.0|13362.0|11330.0|1.0|
|DeltaH_0|-0.000|0.000|-0.000|-0.000|0.000|0.000|31162.0|31101.0|30914.0|34922.0|1.0|
|log_sigma|-14.779|0.168|-15.091|-14.467|0.002|0.002|4880.0|4786.0|4233.0|1562.0|1.0|

### Comparison of 3 PPLs
- Time
- Gelman-rubin statistics 
- Plot mean/std with the functions of the number of samples

