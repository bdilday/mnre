# mnre

Multinomial Random Effects

## Usage

This code solves multinomial mixed effects models. 

## Installing

```
devtools::install_github('bdilday/mnre')
```

## Binomial Example

fit a binomial model using a formula interface

```
library(mnre)
ev = mnre_simulate_multinomial_data_factors(nfct=2, K_class = 2, nlev=50, nobs=20000)
mnre_mod = mnre_fit(y ~ 1 + (1|fct01) + (1|fct02), data=ev$fr, verbose=0)
mnre_mod$theta
[1] 0.9926043 0.9537682
```

fit a binomial model using lower-level `nd_min_fun`

``` 
library(mnre)
ev = mnre_simulate_multinomial_data_factors(nfct=2, K_class = 2, nlev=50, nobs=20000)
nf = mnre::nd_min_fun(ev)
ans = optim(c(1,1), nf, method = "L-BFGS", lower=1e-8)
print(ans$par)
[1] 0.9926043 0.9537682
```

Compare to `lme4`

```
glmer_mod <- glmer(ev$frm, data=ev$fr, family='binomial', nAGQ=0)
glmer_mod@theta
[1] 0.9925264 0.9537615
```

## Multinomial model

Fit a multinomial model

``` 
ev = mnre_simulate_multinomial_data_factors(nfct=1, K_class = 3, nlev=50, nobs=20000)
ev$verbose = 0
nf = mnre::nd_min_fun(ev)
ans = optim(c(1, 1), nf, method = "L-BFGS", lower=1e-8)
print(ans$par)
[1] 0.9351018 1.0758567
```


