
#' @import Matrix
#' @importFrom magrittr %>%
#' @export
mnre_simulate_ev_data <- function(nlim=1000, year=2016, OBP=FALSE) {
  ev_obj <- BProDRA:::generate_model_df(nlim=nlim, year=year)
  dfX <- ev_obj$ev

  # the BPRO code codes it like this
  # ev[cc_bip0,]$outcome <- 1
  # ev[cc_bip1,]$outcome <- 2
  # ev[cc_bip2,]$outcome <- 3
  # ev[cc_hr,]$outcome <- 4
  # ev[cc_so,]$outcome <- 5
  # ev[cc_bb,]$outcome <- 6

  idx1 <- which(dfX$outcome == 1)
  idx2 <- which(dfX$outcome == 2)
  idx3 <- which(dfX$outcome == 3)
  idx4 <- which(dfX$outcome == 4)
  idx5 <- which(dfX$outcome == 5)
  idx6 <- which(dfX$outcome == 6)

  # make batted ball for out the reference
  dfX[idx1,]$outcome <- 0

  # only 1 kind of batted ball for hit
  dfX[c(idx2, idx3),]$outcome <- 1

  # shift the other categories up
  # HR
  dfX[idx4,]$outcome <- 2

  # shift the other categories up
  # SO
  dfX[idx5,]$outcome <- 3

  # shift the other categories up
  # BB
  dfX[idx6,]$outcome <- 4

  # OBP
  if (OBP) {
    dfX[c(idx1, idx5),]$outcome <- 0
    dfX[c(idx2, idx3, idx4, idx6),]$outcome <- 1
  }

  dfX$y <- dfX$outcome

   glf <- lme4::glFormula(y ~ (1|bat_id) + (1|pit_id) + (1|home_team_id),
                          data=dfX, family='binomial')
  #glf <- lme4::glFormula(y ~ (1|pit_id) + (1|bat_id),
   #                      data=dfX, family='binomial')

  dfX <- dfX %>% dplyr::mutate(pid=as.integer(as.factor(pit_id)))
  dfX <- dfX %>% dplyr::mutate(bid=as.integer(as.factor(bat_id)))

  dfX <- dfX %>% dplyr::mutate(sid=as.integer(as.factor(home_team_id)))
  dfX <- dfX %>% dplyr::mutate(bid = bid+max(pid))
  dfX <- dfX %>% dplyr::mutate(sid = sid+max(bid))

  re <- random_effects <- Matrix::t(glf$reTrms$Zt)
  fe <- fixed_effects <- (glf$X)
  k_class <- max(dfX$y)
  beta_re <- matrix(rep(0, k_class * dim(re)[[2]]), ncol=k_class)
  beta_fe <- matrix(rep(0, k_class * dim(fe)[[2]]), ncol=k_class)
  xx <- model.matrix(y ~ bat_id + pit_id + home_team_id, data=dfX)
#  xx <- model.matrix(y ~ bat_id + pit_id, data=dfX)
  y <- matrix(dfX$y, ncol=1)
  theta_init <- matrix(rep(glf$reTrms$theta, k_class), ncol=k_class)
  list(fr=dfX, frm=glf$formula,
       ev_obj=ev_obj, xx=xx, y=y,
       re=re, fe=fe,
       beta_re=beta_re, beta_fe=beta_fe,
       theta_init=theta_init,
       Lind=matrix(glf$reTrms$Lind, ncol=1))
}



#' @export
mnre_simulate_multinomial_data_factors <- function(rseed=101,
                                                           nfct=2,
                                                           nlev=10,
                                                           K_class=3,
                                                           nobs=10000) {

  set.seed(rseed)

  k <- K_class - 1
  fcts <- lapply(1:nfct, function(i) {sprintf("%s%03d", LETTERS[i], 1:nlev)})
  sigmas <- matrix(rep(1, nfct * k), ncol=k)

  coef_null_class <- matrix(rnorm(nfct * length(fcts[[1]])), ncol=nfct)

  # the null class needs to be identically 0, otherwise there will be a
  # spurious corelation between coefficients for the other classes.
  coef_null_class = coef_null_class * 0

  coefs <- lapply(1:nfct, function(i) {
    tmp <- as.data.frame(
      matrix(rnorm(k * length(fcts[[i]])), ncol=k) - coef_null_class[,i]
      )
    tmp$fct <- as.factor(fcts[[i]])
    tmp
  })

  df_obs <- lapply(1:length(coefs), function(i) {
    dplyr::data_frame(fct=sample(fcts[[i]], nobs, replace = TRUE))
  })

  df_obs <- dplyr::bind_cols(df_obs)

  names(df_obs) <- sprintf("fct%02d", 1:nfct)

  dfX <- df_obs
  for (i in 1:length(coefs)) {
    dfX <- dfX %>% merge(coefs[[i]], by.x=sprintf("fct%02d", i), by.y="fct")
  }

  # must be identically 0
  icpt_null_class <- 0
  icpts <- matrix(rnorm(k), ncol=k) - icpt_null_class

  oc <- sapply(1:nrow(dfX), function(i) {

    r <- dfX[i,]
    seq(nfct+1,ncol(r),k)

    lams <- sapply(1:k, function(j) {
      noise <- rnorm(1, 0, 0.2)
      ii=seq(nfct+j,ncol(r),k)
      icpts[[j]] + sum(r[ii]) + noise
    })

    probs <- c(1, exp(lams))

    tmp <- rmultinom(1, 1, probs)
    as.integer(which(tmp > 0) - 1 )
  })

  dfX <- dfX %>% dplyr::mutate(y=oc)
  fct_cut <- which(grepl('^fct|^y$', names(dfX)))

  if (length(fct_cut) > 1) {
    dfY <- dfX[,fct_cut]
  } else {
    dfY <- dplyr::data_frame(fct01=dfX[,fct_cut])
  }

  s = ' y ~ (1|fct01) '

  if (nfct >= 2) {
    for (i in 2:nfct) {
      s <- sprintf("%s + (1|fct%02d) ", s, i)
    }
  }

  frm <- as.formula(s)
  glf <- lme4::glFormula(frm,
                         data=dfY, family='binomial')
  re <- random_effects <- Matrix::t(glf$reTrms$Zt)
  fe <- fixed_effects <- (glf$X)
  k_class <- max(dfY$y)
  beta_re <- matrix(rep(0, k_class * dim(re)[[2]]), ncol=k_class)
  beta_fe <- matrix(rep(0, k_class * dim(fe)[[2]]), ncol=k_class)
  y <- matrix(dfY$y, ncol=1)
  theta_init <- matrix(rep(glf$reTrms$theta, k_class), ncol=k_class)
  ans <- list(true_pars=dfX,
              fr=glf$fr, y=y,
              re=re, fe=fe,
              frm=frm,
              beta_re=beta_re, beta_fe=beta_fe,
              theta_init=theta_init,
              Lind=matrix(glf$reTrms$Lind, ncol=1),
              off_diagonal=0.0)

  ans

}


#' N-dimensional function generator
#' @param ev list
#' @return deviance function
#' @examples 
#' \dontrun{
#' ev = mnre_simulate_multinomial_data_factors(nfct=2, K_class = 2, nlev=50, nobs=20000)
#' nf <- nd_min_fun(ev)
#' nf(c(1,1))
#' }
#' @export
nd_min_fun <- function(ev) {
  
  frm <- ev$frm
  if ("off_diagonal" %in% names(ev)) {
    ev$off_diagonal <- ev$off_diagonal
  } else {
    ev$off_diagonal <- 0.0
  }
  
  if (!"verbose" %in% names(ev)) {
    ev$verbose <- 1
  }
  
  # make sure the data is a data frame, not a tibble
  ev$fr <- as.data.frame(ev$fr)
  
  
  function(mval) {
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

    if (ev$verbose > 0) {
      message(s)      
    }

    
    theta_mat <- matrix(mval, ncol=k_class)
    
    fe2 <- Matrix::Matrix(fe, sparse = TRUE)
    
    if (! "beta_re" %in% names(ev)) {
      ev$beta_re <- matrix(rnorm(ncol(re) * k_class, 0, 0.2), ncol=k_class)
    }

    if (! "beta_fe" %in% names(ev)) {
      ev$beta_fe <- matrix(rnorm(ncol(fe) * k_class, 0, 0.2), ncol=k_class)      
    }    
    
    beta_re <- ev$beta_re
    beta_fe <- ev$beta_fe
    
    if (ev$verbose > 0) {
      message("starting beta ", beta_fe[[1]], " ", beta_re[[1]])
    }
    
    zz <- mnre_fit_sparse(fe2, re, y, theta_mat, Lind, beta_fe, beta_re, verbose = ev$verbose)
    
    ev$beta_fe <<- zz$beta_fixed
    ev$beta_re <<- zz$beta_random
    
    zz$loglk + zz$loglk_det
    
  }
}


#' @export
nd_min_oop_fun <- function(ev) {
  
  frm <- ev$frm
  if ("off_diagonal" %in% names(ev)) {
    ev$off_diagonal <- ev$off_diagonal
  } else {
    ev$off_diagonal <- 0.0
  }
  
  if (!"verbose" %in% names(ev)) {
    ev$verbose <- 1
  }
  
  # make sure the data is a data frame, not a tibble
  ev$fr <- as.data.frame(ev$fr)
  
  mnre_module = Module("mnre_mod", PACKAGE = "mnre", mustStart = TRUE)
  
  
  glf <- lme4::glFormula(ev$frm,
                         data=ev$fr, family='binomial')
  fe <- fixed_effects <- (glf$X)
  re <- random_effects <- Matrix::t(glf$reTrms$Zt)
  fe2 <- Matrix::Matrix(fe, sparse = TRUE)
  
  y <- matrix(ev$fr[,all.vars(ev$frm)[[1]]], ncol=1)
  k_class <- max(y)
  k <- max(y)
  Lind = glf$reTrms$Lind

  if (! "beta_re" %in% names(ev)) {
    ev$beta_re <- matrix(rnorm(ncol(re) * k_class, 0, 0.2), ncol=k_class)
  }
  
  if (! "beta_fe" %in% names(ev)) {
    ev$beta_fe <- matrix(rnorm(ncol(fe) * k_class, 0, 0.2), ncol=k_class)      
  }    
  
  mnre_ptr = new(mnre_module$MNRE, fe2, re)
  
  function(mval) {
    
    s = 'mval '
    for (v in mval) {
      s = sprintf("%s %.4e ", s, v)
    }
    
    if (ev$verbose > 0) {
      message(s)      
    }
    
    theta_mat <- matrix(mval, ncol=k_class)
    
    mnre_ptr$set_theta(theta_mat)
    
    beta_re <- ev$beta_re
    beta_fe <- ev$beta_fe
    
    if (ev$verbose > 0) {
      message("starting beta ", beta_fe[[1]], " ", beta_re[[1]])
    }
    
    zz <- mnre_ptr$mnre_fit_sparse(fe2, re, y, theta_mat, Lind, beta_fe, beta_re, verbose = ev$verbose)
    
    ev$beta_fe <<- zz$beta_fixed
    ev$beta_re <<- zz$beta_random
    
    zz$loglk + zz$loglk_det
    
  }
}


#' @importFrom magrittr %>%
#' @importFrom magrittr %$%
#' @importFrom magrittr %<>%
#'
#' @export
mnre_example_retrosheet_event_data <- function(obp=FALSE) {
  ev <- readRDS(system.file("extdata/events2017.rds", package = "mnre"))
  
  # rm pitchers as hitters
  ev %<>% dplyr::filter(bat_fld_cd != 1)
  
  # anyone with less than 20 PA hitting is generic
  lowpa_batters <- ev %>% 
    dplyr::group_by(bat_id) %>% 
    dplyr::summarise(PA=n()) %>% 
    dplyr::filter(PA<=20) %$% bat_id
  
  # anyone with less than 20 PA hitting is generic
  lowpa_pitchers <- ev %>% 
    dplyr::group_by(pit_id) %>% 
    dplyr::summarise(PA=n()) %>% 
    dplyr::filter(PA<=20) %$% pit_id
  
  if (length(lowpa_batters) > 0) {
    cc <- which(ev$bat_id %in% lowpa_batters)
    ev[cc,]$bat_id <- "xxxxb001"
  }
  
  if (length(lowpa_pitchers) > 0) {
    cc <- which(ev$pit_id %in% lowpa_pitchers)
    ev[cc,]$pit_id <- "xxxxp001"
  }
  
  cc_hr <- which(ev$event_cd == 23)
  cc_so <- which(ev$event_cd == 3)
  cc_bip0 <- which(ev$event_cd == 2)
  cc_bip1 <- which(ev$event_cd == 20)
  cc_bip2 <- which(ev$event_cd >= 21 & ev$event_cd <= 22)
  cc_bb <- which(ev$event_cd >= 14 & ev$event_cd <= 16)
  
  ev$outcome <- NA
  
  if (obp) {
    ev[c(cc_bip0, cc_so),]$outcome <- 0
    ev[c(cc_bip1, cc_bip2, cc_hr, cc_bb),]$outcome <- 1
  } else {
    ev[cc_bip0,]$outcome <- 0
    ev[cc_bip1,]$outcome <- 1
    ev[cc_bip2,]$outcome <- 2
    ev[cc_hr,]$outcome <- 3
    ev[cc_so,]$outcome <- 4
    ev[cc_bb,]$outcome <- 5
  }
  
  ev %<>% dplyr::select(game_id, event_id, event_cd, bat_id, pit_id, home_team_id, outcome)
  
  ev
}


