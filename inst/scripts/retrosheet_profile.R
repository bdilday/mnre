
library(mnre)
library(Rcpp)
library(lme4)

ev = mnre_example_retrosheet_event_data(obp = TRUE)
mnre_mod = mnre_oop_fit(outcome ~ 1 + (1|bat_id) + (1|pit_id), data=ev, verbose=1)
