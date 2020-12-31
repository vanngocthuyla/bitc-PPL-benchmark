Title: "Benchmark of three probabilistic programming languages for Bayesian analysis of isothermal titration calorimetry data"
date: 2020-12-18
author: Van La, Trung Hai Nguyen, Yuanqing Wang, John Chodera, and David D. L. Minh
short: We compared the performance of three probabilistic programming languages - pymc3, pyro, and numpyro - on a binding-related Bayesian statistical inference problem. numpyro was by far the fastest.

Jump to [abstract](#abstract), [introduction](#introduction), [methods](#methods), [results](#results), or [conclusions](#conclusions).

# Abstract

We compared the performance of three probabilistic programming languages - pymc3, pyro, and numpyro - on a binding-related Bayesian statistical inference problem. numpyro was by far the fastest.

# Introduction

The last few years have seen the rapid development of python-based probabilistic programming languages (PPLs) that enable users to specify a Bayesian posterior and perform statistical inference. The PPLs differ in syntax, underlying mathematical libraries, and the use of computing architectures. In the course of a project to develop algorithms and software to analyze data from binding experiments, we sought to compare the performance of a few of them: pymc3, pyro, and numpyro.

We chose these PPLs for various reasons. We selected pymc3 because it syntactically similar to pymc2, which we used in our previous on the analysis of isothermal titration calorimetry (ITC) data ([1](https://github.com/choderalab/bayesian-itc), [2](https://github.com/nguyentrunghai/bayesian-itc/tree/d8cbf43240862e85d72d7d0c327ae2c6f750e600)). We selected pyro based on a suggestion by Theofanis Karaletsos, an early developer of the package, who explained that its open-source developer community is quite active. Theo also suggested that we try numpyro, which is syntactically similar to pyro but based on the numpy mathematical library. It is less established than pyro.

# Methods

## Bayesian posterior

Details of the Bayesian posterior are described in a [scientific journal article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0203224). Some key aspects are:

- Data, D &equiv; {q<sub>1</sub>, q<sub>2</sub>, ..., q<sub>n</sub>}, consists of the observed heats per injection. For this benchmark, we used [Mg1EDTAp1a.DAT](https://github.com/vanngocthuyla/bitc/tree/main/inputs/Mg1EDTAp1a.DAT).
- Parameters, &theta; &equiv; &Delta;G, &Delta;H, &Delta;H<sub>0</sub>, [R]<sub>0</sub>, [L]<sub>s</sub>, log&sigma; are the binding free energy, binding enthalpy, enthalpy of dilution, original receptor concentration in the sample cell, original ligand concentation in the syringe, and a noise parameter.
- Priors are distributed as,
    * &Delta;G ~ Uniform(-40 kcal/mol, 40 kcal/mol),
    * &Delta;H ~ Uniform(-100 kcal/mol, 100k cal/mol),
    * &sigma; ~ uninformative Jeffreys prior,
    * &Delta;H<sub>0</sub> ~ Uniform(q<sub>min</sub> - &Delta;q, q<sub>max</sub> - &Delta;q), where q<sub>min</sub> = min{q<sub>1</sub>, q<sub>2</sub>, ..., q<sub>n</sub>}, q<sub>max</sub> = max{q<sub>1</sub>, q<sub>2</sub>, ..., q<sub>n</sub>} and &Delta;q = q<sub>max</sub> - q<sub>min</sub>,
    * and [R]<sub>0</sub> and [L]<sub>s</sub> follow a lognormal distribution.

## Software installation

PPLs were set up in virtual environments based on the following software and versions:

- Numpyro: Numpyro v0.4.1, Numpy v1.18.5, Matplotlib v3.2.2, Arviz v0.10.0
- Pyro: Pyro v1.5.1, Torch v1.7.0, Numpy v1.18.5, Matplotlib v3.2.2, Arviz v0.10.0
- Pymc3: Pymc3 v3.8, Theano v1.0.5, Pandas v0.25, Arviz v0.4.1

## Sampling from the posterior

Python scripts to set up the Bayesian posterior and perform Markov chain Monte Carlo sampling using the No-U-Turn Sampler were written in
[numpyro](https://github.com/vanngocthuyla/bitc/blob/main/scripts/bitc_numpyro.py), [pyro](https://github.com/vanngocthuyla/bitc/blob/main/scripts/bitc_pyro.py), and [pymc3](https://github.com/vanngocthuyla/bitc/blob/main/scripts/bitc_pymc3.py).
For each PPL, four independent chains with 2000 steps of warmup and 10000 samples were simulated.

The calculations were run using the Pittsburg Supercomputer Center,
on the Bridges cluster, by submitting "Regular Memory" jobs.

- Host: bridges.psc.xsede.org
- CPU Type: Intel Xeon EP-series
- Operating System: CentOS
- Batch System: SLURM
- Memory Per CPU: 128 GB
- CPU Cores Per Node: 28

# Results

## numpyro is orders of magnitude faster than other PPLs

The scripts took the following amount of time to generate an equivalent number of samples:
- pymc3: 425.27 s
- numpyro: 24.01 s
- pyro: 34304.32 s

Among the three PPLs, numpyro is dramatically faster, especially in comparison to pyro, which took about 9.5 hours to generate the same number of samples. Disabling the progress bar decreased the run time of pyro and pymc3, but not by more than 10% of total run time. The difference in speed is stark and remarkable!

Given the contrast in computational expense, we wanted to see if using numpyro entailed any sacrifices to the accuracy or to the effective sample size.

## The sampled posterior distribution is consistent across the PPLs

For all three PPLs, the time series are well-mixed and the distribution is consistent across the repetitions (Figure 1). The estimated mean values and standard deviations of all the parameter are very consistent (Table 1). For all parameters and all PPLs, the Gelman-Rubin statistic (r_hat) was calculated as 1.0, indicating that the independent chains yielded samples from a consistent posterior distribution.

### Figure 1. Histograms and time series of samples generated using pymc3, numpyro, and pyro

- pymc3:
<p><img align="center" src='https://github.com/vanngocthuyla/bitc-PPL-benchmark/raw/main/images/Pymc3_Plot.png' width="800"></p>

- pyro:
<p><img align="center" src='https://github.com/vanngocthuyla/bitc-PPL-benchmark/raw/main/images/Pyro_Plot.png' width="800"></p>

- numpyro:
<p><img align="center" src='https://github.com/vanngocthuyla/bitc-PPL-benchmark/raw/main/images/Numpyro_Plot.png' width="800"></p>

### Table 1. Summary statistics for the posterior distribution of Bayesian parameters

- pymc3

|Parameter|mean|std|
|:-------:|:--:|:-:|
|P0|0.088|0.006|
|Ls|1.119|0.080|
|DeltaG|-8.992|0.076|
|DeltaH|-2.104|0.151|
|DeltaH_0|-0.000|0.000|
|log_sigma|-14.779|0.168|

- pyro

|Parameter|mean|std|
|:-------:|:--:|:-:|
|P0|0.09|0.01|
|Ls|1.12|0.08|
|DeltaG|-8.99|0.08|
|DeltaH|-2.11|0.15|
|DeltaH_0|-0.00|0.00|
|log_sigma|-14.78|0.17|

- numpyro

|Parameter|mean|std|
|:-------:|:--:|:-:|
|P0|0.09|0.01|
|Ls|1.12|0.08|
|DeltaG|-8.99|0.08|
|DeltaH|-2.10|0.15|
|DeltaH_0|-0.00|0.00|
|log_sigma|-14.78|0.16|

## Samples converge at a comparable rate

For all three PPLs, the standard deviation of the estimated mean decreased at a comparable rate as a function of the number of samples (Figure 2). The Gelman-Rubin statistic also decreased to 1 at a comparable rate (Figure 3). Although samples from pymc3 arguably converge more slowly than pyro or numpyro, the relatively small number of chains precludes a more conclusive result.

### Figure 2. Convergence of the mean value of each parameter

For each chain, the mean value of each parameter was estimated as a function of the number of samples. The plots show the mean (left) and standard deviation (right) of these estimates across the independent chains.

<img align="center" src='https://github.com/vanngocthuyla/bitc-PPL-benchmark/raw/main/images/mean_std_P0.png' width="800">
<img align="center" src='https://github.com/vanngocthuyla/bitc-PPL-benchmark/raw/main/images/mean_std_Ls.png' width="800">
<img align="center" src='https://github.com/vanngocthuyla/bitc-PPL-benchmark/raw/main/images/mean_std_DeltaG.png' width="800">
<img align="center" src='https://github.com/vanngocthuyla/bitc-PPL-benchmark/raw/main/images/mean_std_DeltaH.png' width="800">
<img align="center" src='https://github.com/vanngocthuyla/bitc-PPL-benchmark/raw/main/images/mean_std_DeltaH_0.png' width="800">
<img align="center" src='https://github.com/vanngocthuyla/bitc-PPL-benchmark/raw/main/images/mean_std_log_sigma.png' width="800">

### Figure 3. Gelman-Rubin Statistic

The Gelman-Rubin statistic was plotted as a function of the number of samples using Arviz.

<p float="center">
  <img src="https://github.com/vanngocthuyla/bitc-PPL-benchmark/raw/main/images/rhat_P0.png" width="250" />
  <img src="https://github.com/vanngocthuyla/bitc-PPL-benchmark/raw/main/images/rhat_Ls.png" width="250" />
  <img src="https://github.com/vanngocthuyla/bitc-PPL-benchmark/raw/main/images/rhat_DeltaG.png" width="250" />
</p>

<p float="center">
  <img src="https://github.com/vanngocthuyla/bitc-PPL-benchmark/raw/main/images/rhat_DeltaH.png" width="250" />
  <img src="https://github.com/vanngocthuyla/bitc-PPL-benchmark/raw/main/images/rhat_DeltaH_0.png" width="250" />
  <img src="https://github.com/vanngocthuyla/bitc-PPL-benchmark/raw/main/images/rhat_log_sigma.png" width="250" />
</p>

## Conclusions

For the Bayesian analysis of ITC data, numpyro is orders of magnitude faster than pyro and pymc3. There are no apparent caveats to using the PPL.
