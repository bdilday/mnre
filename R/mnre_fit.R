
#' @export 
mnre_fit <- function(frm, data, verbose=0, off_diagonal=0.0) {
  ev <- list() 
  ev$frm <- frm
  ev$fr <- data
  ev$verbose <- verbose
  
  nlev <- length(all.vars(ev$frm))
  mval <- rep(1,  (nlev-1) * max(ev$fr[[all.vars(frm)[1]]]))   

  nf <- nd_min_fun(ev)
  
  ans = optim(mval, nf, method = "L-BFGS", lower=1e-8)

  mnre_fit_to_df(frm, data, ans$par, verbose = verbose, off_diagonal = off_diagonal)
  
}

#' 
mnre_fit_to_df <- function(frm, data, mval, verbose=0, off_diagonal=0.0) {
  data <- as.data.frame(data)
  glf <- lme4::glFormula(frm,
                         data, family='binomial')
  fe <- fixed_effects <- (glf$X)
  re <- random_effects <- Matrix::t(glf$reTrms$Zt)
  
  y <- matrix(data[,all.vars(frm)[[1]]], ncol=1)
  k_class <- max(y)
  k <- max(y)
  Lind = matrix(glf$reTrms$Lind, ncol=1)
  
  theta_mat <- matrix(mval, ncol=k_class)
  covar_mat = mnre_make_covar(theta_mat, Lind, off_diagonal = 0.0)
  left_factor <- mnre_left_covar_factor(covar_mat)
  
  fe_sp <- Matrix::Matrix(fe, sparse = TRUE)

  beta_re <- matrix(rnorm(ncol(re) * k_class), ncol=k_class)
  beta_fe <- matrix(rnorm(ncol(fe) * k_class), ncol=k_class)
  
  zz <- mnre_fit_sparse(fe_sp, re, y, theta_mat, Lind, beta_fe, beta_re, verbose = verbose)
  
  lk1 <- zz$loglk + zz$loglk_det
  bpar = matrix(left_factor %*% matrix(zz$beta_random,ncol=1), ncol=k_class)
  
  lvs <- unlist(sapply(glf$reTrms$flist, levels))
  cc1 <- as.data.frame(cbind(bpar, Lind=Lind))

  ranef_labels <- names(glf$reTrms$cnms)
  df_names <- sapply(1:k_class, function(i) {sprintf("class%02d", i)})
  df_names <- c(df_names, "Lind")
  df_names <- c(df_names, "ranef_label")
  df_names <- c(df_names, "ranef_level")
  
  cc1$ranef <- ranef_labels[cc1[,ncol(cc1)]]
  cc1$lv <- as.vector(matrix(lvs, ncol=1))
  
  mvalX <- t(sapply(1:max(Lind), function(i) {
    idx = which(cc1[,k_class+1] == i)
    tmp <- matrix(bpar[idx,], ncol=k_class)
    apply(tmp, 2, sd)
  }))
  
  names(cc1) <- df_names
  list(ranef=cc1, fixef=zz$beta_fixed, theta=mval)
  
}
