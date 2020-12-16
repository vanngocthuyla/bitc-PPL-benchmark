# Comparison of the three PPLs (numpyro, pyro and pymc3) for running the ITC (isothermal titration calorimetry) data

## 1. Introduction

In previous research ([1](https://github.com/choderalab/bayesian-itc), [2](https://github.com/nguyentrunghai/bayesian-itc/tree/d8cbf43240862e85d72d7d0c327ae2c6f750e600)), MCMC was applied to build Bayesian model that could do sampling from the posterior distribution of thermodynamic parameters from ITC data. 

- Data: D &equiv; {q<sub>1</sub>, q<sub>2</sub>, ..., q<sub>n</sub>} consists of the observed heats per injection
- Parameters: &theta; &equiv; &Delta;G, &Delta;H, &Delta;H<sub>0</sub>, [R]<sub>0</sub>, [L]<sub>s</sub>, log&sigma;)
- Priors: 
<div align="center"> &Delta;G ~ Uniform(-40 kcal/mol, 40 kcal/mol) </div>  

<div align="center"> &Delta;H ~ Uniform(-100 kcal/mol, 100k cal/mol) </div>  

<div align="center"> &sigma; ~ uninformative Jeffreys prior </div>  

<div align="center"> &Delta;H<sub>0</sub> ~ Uniform(q<sub>min</sub> - &Delta;q, q<sub>max</sub> - &Delta;q) </div>  

where q<sub>min</sub> = min{q<sub>1</sub>, q<sub>2</sub>, ..., q<sub>n</sub>}, q<sub>max</sub> = max{q<sub>1</sub>, q<sub>2</sub>, ..., q<sub>n</sub>} and &Delta;q = q<sub>max</sub> - q<sub>min</sub>. Priors for [R]<sub>0</sub>, [L]s follow the lognormal distribution if stated value is available: 
<div align="center"> ln[X]<sub>0</sub> ∼ Normal ([X]<sub>stated</sub>, 0.1∗[X]<sub>stated</sub>) </div>  

Otherwise, they follow the uniform distribution:

<div align="center"> [R]<sub>0</sub> ∼ Uniform(0.001, 1.), [L]<sub>s</sub> ∼ Uniform(0.01, 10.) </div>  
  
Details information about the Bayesian model can be found [here](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0203224). Pymc was used as the probabilistic programming language (PPL) for the model implementation. Now this model can be extended with other two PPLs, which are Numpyro and Pyro. Data for running the Bayesian model and do the comparison can be found here: [Mg1EDTAp1a.DAT](https://github.com/vanngocthuyla/bitc/tree/main/inputs/Mg1EDTAp1a.DAT)

## 2. Installation of three PPLs

- Numpyro: 
Numpyro v0.4.1, Numpy v1.18.5, Matplotlib v3.2.2, Arviz v0.10.0

- Pyro:
Pyro v1.5.1, Torch v1.7.0, Numpy v1.18.5, Matplotlib v3.2.2, Arviz v0.10.0

- Pymc3:
Pymc3 v3.8, Theano v1.0.5, Pandas v0.25, Arviz v0.4.1

- Python scripts: [numpyro](https://github.com/vanngocthuyla/bitc/blob/main/scripts/bitc_numpyro.py), [pyro](https://github.com/vanngocthuyla/bitc/blob/main/scripts/bitc_pyro.py), [pymc3](https://github.com/vanngocthuyla/bitc/blob/main/scripts/bitc_pymc3.py),

## 3. Accessing the PPL models and comparing their performance

### Checking the convergence of 3 PPLs
- Pymc3

<img align="center" src='https://github.com/vanngocthuyla/bitc/blob/main/images/Pymc3_Plot.png' width="800">

|Parameter|mean|sd|hpd_3%|hpd_97%|mcse_mean|mcse_sd|ess_mean|ess_sd|ess_bulk|ess_tail|r_hat|
|:-------:|:--:|:-:|:---:|:-----:|:-------:|:-----:|:------:|:----:|:------:|:------:|:---:|
|P0|0.088|0.006|0.077|0.100|0.000|0.000|12960.0|12866.0|13083.0|8512.0|1.0|
|Ls|1.119|0.080|0.974|1.271|0.001|0.000|12979.0|12887.0|13100.0|8608.0|1.0|
|DeltaG|-8.992|0.076|-9.135|-8.847|0.001|0.000|21974.0|21973.0|21894.0|34408.0|1.0|
|DeltaH|-2.104|0.151|-2.388|-1.826|0.001|0.001|13464.0|13464.0|13362.0|11330.0|1.0|
|DeltaH_0|-0.000|0.000|-0.000|-0.000|0.000|0.000|31162.0|31101.0|30914.0|34922.0|1.0|
|log_sigma|-14.779|0.168|-15.091|-14.467|0.002|0.002|4880.0|4786.0|4233.0|1562.0|1.0|

- Numpyro

<img align="center" width="800" src='https://github.com/vanngocthuyla/bitc/blob/main/images/Numpyro_Plot.png'>

|Parameter|mean|std|median|5.0%|95.0%|n_eff|r_hat|
|:-------:|:--:|:-:|:----:|:--:|:---:|:---:|:---:|
|P0|0.09|0.01|0.09|0.08|0.10|5376.25|1.00|
|Ls|1.12|0.08|1.12|0.98|1.24|5377.76|1.00|
|DeltaG|-8.99|0.08|-8.99|-9.12|-8.87|7859.73|1.00|
|DeltaH|-2.10|0.15|-2.10|-2.34|-1.86|5488.17|1.00|
|DeltaH_0|-0.00|0.00|-0.00|-0.00|-0.00|11006.09|1.00|
|log_sigma|-14.78|0.16|-14.79|-15.05|-14.52|8316.83|1.00

- Pyro

<img align="center" src='https://github.com/vanngocthuyla/bitc/blob/main/images/Pyro_Plot.png' width="800">

|Parameter|mean|std|median|5.0%|95.0%|n_eff|r_hat|
|:-------:|:--:|:-:|:----:|:--:|:---:|:---:|:---:|
|P0|0.09|0.01|0.09|0.08|0.10|9944.88|1.00|
|Ls|1.12|0.08|1.11|0.99|1.24|9950.80|1.00|
|DeltaG|-8.99|0.08|-8.99|-9.12|-8.87|14202.63|1.00|
|DeltaH|-2.11|0.15|-2.10|-2.35|-1.86|10039.88|1.00|
|DeltaH_0|-0.00|0.00|-0.00|-0.00|-0.00|20708.19|1.00|
|log_sigma|-14.78|0.17|-14.79|-15.05|-14.51|16259.00|1.00|

The trace plots and r_hat (Gelman-rubin) factors above indicate that in each PPL, the NUTS sampling model converged. To confirm that there was no difference between three Bayesian models, some statistical metrics would be plotted with the functions of the number of samples. In additions, the time for running was accessed to decide which PPL could provide the better performance. 

### Comparison of 3 PPLs

#### Plot mean/std with the functions of the number of samples

Calculate the mean and standard deviation of 8-chain samples and plot with the function of the number of samples

<img align="center" src='https://github.com/vanngocthuyla/bitc/blob/main/images/mean_std_P0.png' width="800">
<img align="center" src='https://github.com/vanngocthuyla/bitc/blob/main/images/mean_std_Ls.png' width="800">
<img align="center" src='https://github.com/vanngocthuyla/bitc/blob/main/images/mean_std_DeltaG.png' width="800">
<img align="center" src='https://github.com/vanngocthuyla/bitc/blob/main/images/mean_std_DeltaH.png' width="800">
<img align="center" src='https://github.com/vanngocthuyla/bitc/blob/main/images/mean_std_DeltaH_0.png' width="800">
<img align="center" src='https://github.com/vanngocthuyla/bitc/blob/main/images/mean_std_log_sigma.png' width="800">

Even though the means from some parameters (P0, Ls and DeltaH) of Pyro model were little different to those of Numpyro and Pymc3 models, this difference was relative small and could be ignored. For the standard deviations, except for log_sigma, the standard deviations of other parameters calculated by three PPLs were approximate. This suggested that the Bayesian models of three PPLs reached to the similar convergence. 

#### Gelman-rubin statistics 

Use function from Arviz to calculate r_hat factor of each paramete and plot r_hat with the function of the number of samples

<p float="center">
  <img src="https://github.com/vanngocthuyla/bitc/blob/main/images/rhat_P0.png" width="300" />
  <img src="https://github.com/vanngocthuyla/bitc/blob/main/images/rhat_Ls.png" width="300" />
  <img src="https://github.com/vanngocthuyla/bitc/blob/main/images/rhat_DeltaG.png" width="300" />
</p>

<p float="center">
  <img src="https://github.com/vanngocthuyla/bitc/blob/main/images/rhat_DeltaH.png" width="300" />
  <img src="https://github.com/vanngocthuyla/bitc/blob/main/images/rhat_DeltaH_0.png" width="300" />
  <img src="https://github.com/vanngocthuyla/bitc/blob/main/images/rhat_log_sigma.png" width="300" /> 
</p>

r_hat (Gelman-rubin) factor is a common factor that can often be used to as the diagnosis for the convergence of the Bayesian model. From the above plots, except for r_hat calculated from the sampling of log_sigma of Pymc3 model was little different to those of Numpyro and Pyro models, the r_hat factors of other parameters from three PPLs were nearly equal to 1, pointing out that there was no difference between the multiple Markov chains of each PPL.

#### Time

Time for running 4 chains of 2000 warmups and 10000 samples by NUTS sampling: 
- pymc3: 425.27 s
- numpyro: 24.01 s
- pyro: 34304.32 s

Among the three PPLs, numpyro dramatically took the least time for running, especially in comparison to pyro, which took about 9.5 hours to reach the similar convergence as numpyro. 

Note: disable the progressbar while running can decrease the running time of pyro and pymc3, but not decrease more than 10% of total running time for each of these two PPLs. 


## Conclusion

Numpyro can be considered as the fastly-performed PPL in comparison to Pyro and Pymc3. 
