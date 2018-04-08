
#' @import Rcpp
#' @importFrom Rcpp cpp_object_initializer
#' @export
#' 
mnre_oop_fit = function(ev) {
  mnre_mod_cpp = Rcpp::Module("mnre_mod", PACKAGE = "mnre", mustStart = TRUE)
  
  mval = c(1,1)
  glf <- lme4::glFormula(ev$frm,
                         data=ev$fr, family='binomial')
  fe <- fixed_effects <- (glf$X)
  re <- random_effects <- Matrix::t(glf$reTrms$Zt)
  
  y <- matrix(ev$fr[,all.vars(ev$frm)[[1]]], ncol=1)
  k_class <- max(y)
  k <- max(y)
  Lind = glf$reTrms$Lind
  
  s = 'mval '
  for (v in mval) {
    s = sprintf("%s %.4e ", s, v)
  }
  
  if (!"verbose" %in% names(ev)) {
    ev$verbose = 0
  }
  
  if (ev$verbose > 0) {
    message(s)      
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
  
  mnre_ptr = new(mnre_mod_cpp$MNRE, fe2, re)
  
  mnre_ptr$set_y(y)
  mnre_ptr$set_lind(Lind)
  mnre_ptr$set_beta(beta_fe, beta_re)
  mnre_ptr$set_k_class(k_class)
  mnre_ptr$set_dims()
  
  mnre_oop_fit_fun = function(mval) {
    theta_mat <- matrix(mval, ncol=k_class)
    mnre_ptr$set_theta(theta_mat)
    zz <- mnre_ptr$mnre_oop_fit()   
    zz$loglk + zz$loglk_det
  }
  
}
