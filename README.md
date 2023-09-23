# Gravity Assignment 3

  * 30% of final grade
  * assigned 29 Sep 2023
  * due 13 Oct 2023

---

## Problem 1

> There are snippets of simulated data for a hypothetical into which we will inject a few signals.
> Your job is to write a matched-filter search from scratch to detect these signals.
>
> Write a matched filter search and use it to determine
> 
>   * the number of signals present
>   * the statistical significance of each signal (i.e., a FAR)
>   * the physical amplitude, signal to noise ratio, and reference time for each detected signal
>
> Signals will be sine-Gaussians of the form
> 
> ```math
> h(t) = A \cos(2\pi f_o (t-t_o) + \phi_o) \exp\left( -\frac{(t-t_o)^2}{2\tau^2} \right)
> ```
> 
> Additionally, you can assume that the noise is stationary, Gaussian, and white.

Simulated data was generated via (this script is contained within Gravity-Assignment-3)

```
./make-data \
    --seed 123456789 \
    --duration 4096 \
    --sample-rate 512 \
    --noise-sigma 1.0 \
    --signal-rate 0.005 \
    --prior A 0.1 5.0 \
    --prior fo 20.0 \
    --prior tau 2.0 \
    --verbose
```

and we can see the signals in the data below:

<img src="assignment-3.png">

We run our search via
```
./search \
    assignment-3.hdf \
    --verbose
```

which produces

**WRITE THIS**
