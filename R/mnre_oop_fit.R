
#' @import Rcpp
#' @importFrom Rcpp cpp_object_initializer
#' @importFrom Rcpp evalCpp
#' @export
#' 
nd_oop_min_fun = function(ev) {
  mnre_mod_cpp = Rcpp::Module("mnre_mod", PACKAGE = "mnre", mustStart = TRUE)
  
  glf <- lme4::glFormula(ev$frm,
                         data=ev$fr, family='binomial')
  fe <- fixed_effects <- (glf$X)
  re <- random_effects <- Matrix::t(glf$reTrms$Zt)
  
  y <- matrix(ev$fr[,all.vars(ev$frm)[[1]]], ncol=1)
  k_class <- max(y)
  k <- max(y)
  Lind = glf$reTrms$Lind
  
  if (!"verbose" %in% names(ev)) {
    ev$verbose = 0
  }

  
  fe2 <- Matrix::Matrix(fe, sparse = TRUE)
  
  if (! "beta_re" %in% names(ev)) {
    ev$beta_re <- matrix(rnorm(ncol(re) * k_class, 0, 0.2), ncol=k_class)
  }
  
  if (! "beta_fe" %in% names(ev)) {
    ev$beta_fe <- matrix(rnorm(ncol(fe) * k_class, 0, 0.2), ncol=k_class)      
  }    
  
  beta_re <- ev$beta_re
  beta_fe <- ev$beta_fe
  
  mnre_ptr = new(mnre_mod_cpp$MNRE, fe2, re, ev$verbose)
  
  mnre_ptr$set_y(y)
  mnre_ptr$set_lind(Lind)
  mnre_ptr$set_beta(beta_fe, beta_re)
  mnre_ptr$set_k_class(k_class)
  mnre_ptr$set_dims()
  
  mnre_oop_fit_fun = function(mval) {
    if (ev$verbose > 0) {
      s = 'mval '
      for (v in mval) {
        s = sprintf("%s %.4e ", s, v)
      }  
      s = sprintf("%s %.4e ", s, ev$beta_re[1])
      message(s)      
    }
    
    theta_mat <- matrix(mval, ncol=k_class)
    mnre_ptr$set_theta(theta_mat)
    zz <- mnre_ptr$mnre_oop_fit()   
    
    ev$beta_fe <<- zz$beta_fixed
    ev$beta_re <<- zz$beta_random
    mnre_ptr$set_beta(ev$beta_fe, ev$beta_re)
    
    zz$loglk + zz$loglk_det
  }
  
}

#' @export 
mnre_oop_fit <- function(frm, data, verbose=0, off_diagonal=0.0) {
  ev <- list() 
  ev$frm <- frm
  ev$fr <- data
  ev$verbose <- verbose
  
  nlev <- length(all.vars(ev$frm))
  mval <- rep(1,  (nlev-1) * max(ev$fr[[all.vars(ev$frm)[1]]]))   

  nf <- nd_oop_min_fun(ev)
  
  ans = optim(mval, nf, method = "L-BFGS", lower=1e-8)
  
  mnre_fit_to_df(frm, data, ans$par, verbose = verbose, off_diagonal = off_diagonal)
  
}
