
#' @export 
mnre_fit <- function(frm, data, verbose=0, off_diagonal=0.0) {
  ev <- list() 
  ev$frm <- frm
  ev$fr <- data
  ev$verbose <- verbose
  
  nlev <- length(all.vars(ev$frm))
  mval <- rep(1,  (nlev-1) * max(ev$fr$y))   

  nf <- nd_min_fun(ev)
  
  ans = optim(mval, nf, method = "L-BFGS", lower=1e-8)

}