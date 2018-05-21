
nd_min_fun <- function(ev) {

  frm <- ev$frm
  if ("off_diagonal" %in% names(ev)) {
    ev$off_diagonal <- ev$off_diagonal
  } else {
    ev$off_diagonal <- 0.0
  }
  
  if ("verbose" %in% names(ev)) {
    ev$verbose <- ev$verbose
  } else {
    ev$verbose <- 1
  }

  nd_min8 <- function(mval) {
    glf <- lme4::glFormula(ev$frm,
                           data=ev$fr, family='binomial')
    fe <- fixed_effects <- (glf$X)
    re <- random_effects <- Matrix::t(glf$reTrms$Zt)
    y <- matrix(ev$y, ncol=1)
    k_class <- max(y)
    k <- max(y)
    Lind = ev$Lind

    s = 'mval '
    for (v in mval) {
      s = sprintf("%s %.4e ", s, v)
    }
    
    if (ev$verbose > 0) {
      message(s)
    }

    theta_mat <- matrix(mval, ncol=k_class)

    fe2 <- Matrix::Matrix(fe, sparse = TRUE)

    beta_re <- ev$beta_re
    beta_fe <- ev$beta_fe

    # beta_re <- matrix(rep(0, ncol(re) * k_class), ncol=k_class)
    # beta_fe <- matrix(rep(0, ncol(fe) * k_class), ncol=k_class)
    #
    
    if (ev$verbose > 0) {
      message("starting beta ", beta_fe[[1]], " ", beta_re[[1]])
    }
    
    zz <- mnre_fit_sparse(fe2, re, y, theta_mat, Lind, beta_fe, beta_re, verbose = ev$verbose)

    ev$beta_fe <<- zz$beta_fixed
    ev$beta_re <<- zz$beta_random

    zz$loglk + zz$loglk_det

  }

  nd_min8a <- function(mval) {
    glf <- lme4::glFormula(ev$frm,
                           data=ev$fr, family='binomial')
    fe <- fixed_effects <- (glf$X)
    re <- random_effects <- Matrix::t(glf$reTrms$Zt)
    y <- matrix(ev$y, ncol=1)
    k_class <- max(y)
    k <- max(y)
    Lind = ev$Lind

    s = 'mval '
    for (v in mval) {
      s = sprintf("%s %.4e ", s, v)
    }
    message(s)

    theta_mat <- matrix(mval, ncol=k_class)

    fe2 <- Matrix::Matrix(fe, sparse = TRUE)

    beta_re <- ev$beta_re
    beta_fe <- ev$beta_fe

    # beta_re <- matrix(rep(0, ncol(re) * k_class), ncol=k_class)
    # beta_fe <- matrix(rep(0, ncol(fe) * k_class), ncol=k_class)
    #
    message("starting beta ", beta_fe[[1]], " ", beta_re[[1]])
    zz <- mnre_fit_sparse(fe2, re, y, theta_mat, Lind, beta_fe, beta_re)

    ev$beta_fe <<- zz$beta_fixed
    ev$beta_re <<- zz$beta_random

    list(loglk=zz$loglk, loglk_det=zz$loglk_det)

  }


  nd_min9 <- function(mval_mult) {
    mval0 <- ev$mval0
    mval <- mval0 * mval_mult

    glf <- lme4::glFormula(ev$frm,
                           data=ev$fr, family='binomial')
    fe <- fixed_effects <- (glf$X)
    re <- random_effects <- Matrix::t(glf$reTrms$Zt)
    y <- matrix(ev$y, ncol=1)
    k_class <- max(y)
    k <- max(y)
    Lind = ev$Lind

    s = 'mval '
    for (v in mval) {
      s = sprintf("%s %.4e ", s, v)
    }
    message(s)

    theta_mat <- matrix(mval, ncol=k_class)

    fe2 <- Matrix::Matrix(fe, sparse = TRUE)
    zz <- mnre_fit_sparse(fe2, re, y, theta_mat, Lind)

    zz$loglk + zz$loglk_det

  }


  list(nd_min8, nd_min9,nd_min8a)

}

generate_obp_sid_comp <- function(mseq=seq(1, 1000, 30)) {
  ev <- lmearmadillo_simulate_obp_data_sid(nlim=NULL)
  lmer_mod <- glmer(outcome ~ 1 + (1|sid), data=ev$ev, nAGQ=0, family='binomial')
  glf <- lme4::glFormula(outcome ~ (1|HOME_TEAM_ID), data=ev$fr)
  fe <- fixed_effects <- (glf$X)
  re <- random_effects <- Matrix::t(glf$reTrms$Zt)
  y <- matrix(ev$y, ncol=1)
  mm = matrix(rep(1, 1 * dim(ev$re)[[2]]), ncol=1)
  multinom_run_test <- function(ev = NULL) {
    if (is.null(ev)) {
      ev <-lmearmadillo_simulate_multinomial_data_factors(nfct = 2,
                                                          nlev = 3,
                                                          K_class = 3)
    }
    
    mval <- c(2, 3, 5, 7)
    glf <- lme4::glFormula(ev$frm,
                           data=ev$fr, family='binomial')
    fe <- fixed_effects <- (glf$X)
    re <- random_effects <- Matrix::t(glf$reTrms$Zt)
    y <- matrix(ev$y, ncol=1)
    k_class <- max(y)
    k <- max(y)
    Lind = ev$Lind
    
    theta_mat <- matrix(mval, ncol=k_class)
    
    fe_sp <- Matrix::Matrix(fe, sparse = TRUE)
    fe2 <- cbind(fe_sp, fe_sp)
    re2 <- cbind(re, re)
    
    beta_random <- matrix(rep(0, ncol(re) * k_class), ncol=k_class)
    beta_fixed <- matrix(rep(0, ncol(fe) * k_class), ncol=k_class)
    
    mu <- mnre_mu_x(fe2, re2, beta_fixed+1, beta_random)
    
    cc = mnre_make_covar(theta_mat, Lind, off_diagonal = 0.0)
    left_factor <- mnre_left_covar_factor(cc)
    zlam = re2 %*% left_factor
    
    ftwf = fill_mtwm_4(fe_sp, fe_sp, mu, matrix(0, ncol=1))
    ftwr = fill_mtwm_4(fe_sp, re, mu, matrix(0, ncol=1))
    rtwr = fill_mtwm_4(re, re, mu, matrix(1, ncol=1))
    
    lhs_mat <- rbind(cbind(ftwf, ftwr), cbind(Matrix::t(ftwr), rtwr))
    
    dy <- matrix(rep(0, nrow(mu) * k_class), ncol=k_class)
    for (i in 1:nrow(mu)) {
      iy = as.integer(y[i]);
      for (k in 1:k_class) {
        if (iy == k+1) {
          dy[i, k] = (1 - mu[i, k]);
        } else {
          dy[i, k] = (0 - mu[i, k]);
        }
      }
    }
    
    rhs_fe = Matrix::t(fe) %*% dy
    rhs_re = Matrix::t(re) %*% dy
    
    n_dim_random = nrow(rhs_re)
    for (d1 in 1:n_dim_random) {
      for (k1 in 1:k_class) {
        rhs_re[d1, k1] = rhs_re[d1, k1] - beta_random[d1, k1]
      }
    }
    
    rhs_mat <- rbind(matrix(rhs_fe, ncol=1), matrix(rhs_re, ncol=1))
    
  }
  
  Lind = ev$Lind

  zz <- with(ev, mnre_step(fe, re, y, beta_fe*0, beta_re*0, mm, Lind))
  ll = list()

  for (ival in seq_along(mseq)) {
    mval = mseq[ival]
    message(mval)
    i = 0
    brd = 100
    mm = matrix(rep(mval, 1 * dim(ev$re)[[2]]), ncol=1)
    while ( (mean(brd**2) > 1e-15) && (i < 10) ) {
      i <- i + 1
      beta_fe <- zz$beta_fixed
      beta_re <- zz$beta_random
      zz <- mnre_step(fe, re, y, beta_fe, beta_re, mm, Lind)
      brd <- zz$beta_random_diff
    }

    beta_fe <- zz$beta_fixed
    beta_re <- zz$beta_random

    mu <- mnre_mu(fe, re, beta_fe, beta_re)
    rtwr = r_based_fill_mtwm_2(re, re, mu, mm)
    logdet = Matrix::determinant(rtwr, logarithm = TRUE)
    logdet_theta = Matrix::determinant(Matrix::Diagonal(mm, n=nrow(mm)), logarithm = TRUE)

    tmp = list(mval=mval,
               loglk=zz$loglk,
               logdet=logdet$modulus[[1]],
               logdet_theta=logdet_theta$modulus[[1]],
               i=i)
    ll[[ival]] = tmp
  }

  ll
}


#' @export
glmer_mods <- function(ev) {

  mods <- list()
  for (i in 1:max(ev$y)) {
    message("fitting model for ", i)
    tmp <- ev$fr
    cc <- which(ev$fr$y == i)
    tmp[cc,]$y <- 1
    tmp[-cc,]$y <- 0
    glmer_mod <- glmer(ev$frm, data=tmp, family='binomial', nAGQ=0)
    mods[[i]] <- glmer_mod
    rm(tmp)
  }

  mods


}

