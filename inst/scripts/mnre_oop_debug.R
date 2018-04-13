
library(mnre)
library(Rcpp)
library(lme4)

ev = mnre_simulate_multinomial_data_factors(nfct=2, K_class = 2, nlev=50, nobs=20000)
ans = mnre_oop_fit(ev$frm, ev$fr, verbose = 1)

glmer_mod = glmer(ev$frm, data=ev$fr, family='binomial', nAGQ = 0)