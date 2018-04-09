
#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include "mnre.h"
#include "mnre_types.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp11)]]


using namespace arma;

#define DEBUG 1
#define EXPAND_ROW 0
#define EXPAND_COLUMN 1

void MNRE::mnre_echo(NumericVector &x) {
  for (int i=0; i < x.size(); i++) {
    Rcpp::Rcout << i << " " << x[i] << " " << std::endl;
  }
}


//' @export
// [[Rcpp::export]]
MNRE::MNRE(const arma::sp_mat& fixed_effects,
           const arma::sp_mat& random_effects
             ): 
  fixed_effects (fixed_effects),
  random_effects (random_effects)
  {
  Rcpp::Rcout << "I am mnre!" << std::endl;
};

//' @export
// [[Rcpp::export]]
MNRE::MNRE(const arma::sp_mat& fixed_effects,
           const arma::sp_mat& random_effects,
           int verbose_
): 
  fixed_effects (fixed_effects),
  random_effects (random_effects)
{
  Rcpp::Rcout << "I am mnre!" << std::endl;
  verbose = verbose_;
};

//' @export
// [[Rcpp::export]]
MNRE::MNRE() {
  Rcpp::Rcout << "I am mnre!" << std::endl;
};

//' @export
// [[Rcpp::export]]
MNRE::MNRE(int x) {
  Rcpp::Rcout << "I am mnre! " << x << std::endl;
};


//' @export
// [[Rcpp::export]]
SEXP mnre_create_empty() {
  MNRE *m = new MNRE();
  return wrap(XPtr<MNRE>(m, true));
}

//' @export
// [[Rcpp::export]]
SEXP mnre_create(const arma::sp_mat fixed_effects,
                 const arma::sp_mat random_effects,
                 const arma::vec& y,
                 const arma::mat& theta_norm,
                 const arma::uvec& Lind
                   ) {
  MNRE *m = new MNRE(fixed_effects,
                     random_effects
                    );
  return wrap(XPtr<MNRE>(m, true));
}

arma::mat MNRE::mnre_mu_x_oop(const arma::sp_mat &fe_x,
                              const arma::sp_mat &re_x,
                              const arma::mat &beta_fixed,
                              const arma::mat &beta_random) {
  
  // int n_data = fe_x.n_rows;
  // int k_class = beta_fixed.n_cols;
  
  int ndim_fe = fe_x.n_cols / k_class;
  int ndim_re = re_x.n_cols / k_class;
  
  arma::sp_mat fixed_effects(n_data, ndim_fe);
  arma::sp_mat random_effects(n_data, ndim_re);
  
  arma::mat mu(n_data, k_class);
  arma::mat eta(n_data, k_class);
  arma::mat ee(n_data, k_class);
  
  arma::vec denom(n_data);
  
  for (int k=0; k < k_class; k++) {
    fixed_effects = fe_x.submat(0, ndim_fe * k, n_data-1, ndim_fe * (k+1) - 1);
    random_effects = re_x.submat(0, ndim_re * k, n_data-1, ndim_re * (k+1) - 1);
    eta.col(k) = fixed_effects * beta_fixed.col(k) + random_effects * beta_random.col(k);
    ee.col(k) = exp(eta.col(k));
  }
  
  for (int i=0; i< n_data; i++) {
    denom(i) = 1.0;
    for (int k=0; k < k_class; k++) {
      denom(i) += ee(i, k);
    }
    
    for (int k=0; k < k_class; k++) {
      mu(i, k) = ee(i, k)/denom(i);
    }
  }
  
  return mu;
}

arma::sp_mat MNRE::mnre_left_covar_factor_oop(arma::sp_mat& x1) {
  arma::sp_mat left_factor;
  
  SimplicialLLT<SparseMatrix <double> > solver;
  
  SparseMatrix<double> rx = Rcpp::as<SparseMatrix <double> >(wrap(x1));
  solver.compute(rx);
  left_factor = Rcpp::as<arma::sp_mat>(wrap(solver.matrixL()));
  return left_factor;
}


arma::sp_mat MNRE::mnre_oop_expand_matrix(const arma::sp_mat& x1, int k_class, int direction) {
  
  arma::sp_mat expanded_mat;
  
  for (int k1=0; k1 < k_class; k1++) {
    if (k1 == 0) {
      expanded_mat = x1;
    } else {
      if (direction == EXPAND_ROW) {
        expanded_mat = arma::join_vert(expanded_mat, x1);
      } else if (direction == EXPAND_COLUMN) {
        expanded_mat = arma::join_horiz(expanded_mat, x1);
      } else {
        throw std::invalid_argument("direction must be EXPAND_ROW or EXPAND_COLUMN");
      }
    }
  }
  return expanded_mat;
}

Rcpp::List MNRE::mnre_oop_fit() {

  arma::mat lambda_ones(theta_mat.n_rows, theta_mat.n_cols);
  lambda_ones.fill(1);
  
  SimplicialLLT<SparseMatrix <double> > mtwm_solver;
    
  int step_counter;
  double tol = 1e-5;
  double grad_tol = 1e-5;
  
  double loglk_det;
  double loglk_det_val;
  double loglk_det_sign;
  
  Rcpp::List ll;
  int half_steps = 128;
  double newton_step_factor = 1.0;
  double lk_diff_stat, loglk_old, lk_new, beta_diff;
  int max_step = 200;
  
  arma::sp_mat covar_mat = mnre_make_covar_oop(0.0);
  arma::mat mu;
  arma::sp_mat left_factor = mnre_left_covar_factor_oop(covar_mat);
  
  arma::sp_mat re_x = mnre_oop_expand_matrix(random_effects, k_class, EXPAND_COLUMN);
  arma::sp_mat fe_x = mnre_oop_expand_matrix(fixed_effects, k_class, EXPAND_COLUMN);
  
  if (DEBUG) {
    Rcpp::Rcout << " left_factor " << left_factor.n_rows << " " << left_factor.n_cols << " " << left_factor(0, 0) << std::endl;
    Rcpp::Rcout << " random_effects " << random_effects.n_rows << " " << random_effects.n_cols << std::endl;
    Rcpp::Rcout << " re_x " << re_x.n_rows << " " << re_x.n_cols << " " << re_x(1, 1) << std::endl;
  }
  
  arma::sp_mat ZLam = re_x * left_factor;
  
  if (DEBUG) {
    Rcpp::Rcout << " ZLam " << ZLam.n_rows << " " << ZLam.n_cols << " " << ZLam(1, 2) << std::endl;
  }
  
  arma::sp_mat rtwr;
  arma::sp_mat sp_diag_mat = speye(n_dim_random * k_class, n_dim_random * k_class);

  SparseMatrix<double> rx;
  
  for (int istep=0; istep < max_step; istep++) {
    if (DEBUG) {
      Rcpp:Rcout << " istep " << istep << std::endl;
    }
    
    mu = mnre_mu_x_oop(fe_x, ZLam, mnre_beta_fixed, mnre_beta_random);
    
    if (DEBUG) {
      Rcpp::Rcout << " call step... " << std::endl;
    }
    
    ll = mnre_step_oop();
    
    if (DEBUG) {
      Rcpp::Rcout << " call step... DONE " << std::endl;
    }
    
    loglk_old = (double)(ll["loglk_old"]);
    lk_diff_stat = (double)(ll["loglk"]) - loglk_old;
    
    if (verbose >=1) {
      Rcpp::Rcout << " istep " << istep <<
        " lk_diff_stat " << lk_diff_stat <<
          " lk value " << (double)ll["loglk"] <<
            " grad_check " << (double)ll["grad_check"] << std::endl;
    }
    
    if (lk_diff_stat > 0) { // fit got worse
      step_counter = 0;
      while( (step_counter < half_steps) && ( lk_diff_stat >= 0 ) ) {
        newton_step_factor = 0.5 * newton_step_factor;
        lk_new =
          mnre_lk_oop(mnre_beta_fixed + newton_step_factor * as<arma::mat>(ll["beta_fixed_diff"]),
                      mnre_beta_random + newton_step_factor * as<arma::mat>(ll["beta_random_diff"]),
                      y);
        lk_diff_stat = lk_new - loglk_old;
        step_counter += 1;
        
        if (DEBUG) {
          Rcpp::Rcout << " istep " << istep << " half-step " <<  step_counter <<
            " lk_diff_stat " << lk_diff_stat <<
              " lk_diff_stat / step " << lk_diff_stat / newton_step_factor <<
                " grad_check " << (double)ll["grad_check"] << std::endl;
        }
      }
    }
    
    Rcpp::List ans;
    if (lk_diff_stat >= 0) { // fit got worse
      rtwr = fill_mtwm_x_oop(ZLam, ZLam, mu);
      rtwr = rtwr + sp_diag_mat;
      
      // TODO: either use the left factor or use rtwr
      rx = Rcpp::as<SparseMatrix <double> >(wrap(rtwr));
      mtwm_solver.compute(rx);
      left_factor = Rcpp::as<arma::sp_mat>(wrap(mtwm_solver.matrixL()));
      //loglk_det = arma::sum(std::log(left_factor.diag()));
      //loglk_det = 0;
      arma::log_det(loglk_det_val, loglk_det_sign, arma::mat(rtwr));
      loglk_det = loglk_det_val * loglk_det_sign;
      
      if (verbose >= 1) {
        Rcpp::Rcout << "half step cycle did not find a better fit" << std::endl;
      }
      
      ans["beta_fixed"] = mnre_beta_fixed;
      ans["beta_random"] = mnre_beta_random;
      ans["loglk"] = (double)(ll["loglk"]);
      ans["loglk_det"] = loglk_det;
      
      return ans;
    } else {
      mnre_beta_fixed = mnre_beta_fixed + newton_step_factor * as<arma::mat>(ll["beta_fixed_diff"]);
      mnre_beta_random = mnre_beta_random + newton_step_factor * as<arma::mat>(ll["beta_random_diff"]);
      newton_step_factor = 1.0;
    }
    
    if (std::abs(lk_diff_stat) <= tol && (double)ll["grad_check"] <= grad_tol) {
      rtwr = fill_mtwm_x_oop(ZLam, ZLam, mu);
      rtwr = rtwr + sp_diag_mat;
      // TODO: either use the left factor or use rtwr
      rx = Rcpp::as<SparseMatrix <double> >(wrap(rtwr));
      mtwm_solver.compute(rx);
      left_factor = Rcpp::as<arma::sp_mat>(wrap(mtwm_solver.matrixL()));
      //loglk_det = arma::sum(std::log(left_factor.diag()));
      arma::log_det(loglk_det_val, loglk_det_sign, arma::mat(rtwr));
      loglk_det = loglk_det_val * loglk_det_sign;
      
      ans["beta_fixed"] = mnre_beta_fixed;
      ans["beta_random"] = mnre_beta_random;
      ans["loglk"] = (double)(ll["loglk"]);
      ans["loglk_det"] = loglk_det;
      return ans;
    }
    
  } // steps
  
} // end function


void MNRE::set_dims() {
  n_data = y.n_elem;  
  n_dim_fixed = mnre_beta_fixed.n_rows;
  n_dim_random = mnre_beta_random.n_rows;
  n_dim = n_dim_fixed + n_dim_random;
  D_times_K = n_dim * k_class;
  Dfixed_times_K = n_dim_fixed * k_class;
  Drandom_times_K = n_dim_random  * k_class;
}

arma::sp_mat MNRE::fill_mtwm_x_oop(const arma::sp_mat& x1, const arma::sp_mat& x2,
                             const arma::mat& mu) {
  
  int Kclass = mu.n_cols;
  
  int D1 = x1.n_cols / Kclass;
  int D2 = x2.n_cols / Kclass;
  
  int D1_times_K = D1 * Kclass;
  int D2_times_K = D2 * Kclass;
  arma::sp_mat mtwm(D1_times_K, D2_times_K);
  arma::vec ww;
  int idx1, idx2;
  double aa;
  arma::sp_mat tmp_mat;
  int idx11, idx12, idx21, idx22;
  arma::sp_mat ww_sp;
  int tmp;
  
  arma::mat ww_con1(x1.n_rows, D1);
  arma::mat ww_con2(x2.n_rows, D2);
  ww_con1.fill(1);
  ww_con2.fill(1);
  arma::mat ww_mat(mu.n_rows, Kclass);
  
  for (int k1=0; k1 < Kclass; k1++) {
    for (int k2=0; k2 < Kclass; k2++) {
      
      if (k1 == k2) {
        ww =  mu.col(k1) % (1-mu.col(k1));
      } else {
        ww = mu.col(k1) % mu.col(k2);
      }
      ww = arma::sqrt(ww);
      for (int i=0; i<ww_con1.n_cols; i++) {
        ww_con1.col(i) = ww;
      }
      for (int i=0; i<ww_con2.n_cols; i++) {
        ww_con2.col(i) = ww;
      }
      
      idx11 = (k1)*D1 ;
      idx12 = idx11 + D1 - 1;
      idx21 = (k2)*D2 ;
      idx22 = idx21 + D2 - 1;
      
      if (k1 == k2) {
        tmp_mat = (x1.cols(idx11, idx12) % ww_con1).t() * (x2.cols(idx21, idx22) % ww_con2);
      } else {
        tmp_mat = -(x1.cols(idx11, idx12) % ww_con1).t() * (x2.cols(idx21, idx22) % ww_con2);
      }
      
      mtwm( arma::span(idx11, idx12), arma::span(idx21, idx22) ) = tmp_mat;
      
    }
  }
  
  return mtwm;
}


Rcpp::List MNRE::mnre_step_oop() {
  // The left factor of the covariance should already be applied to the random effects
  // TODO: if so, then lambda_norm must be 1; should that be enforced?
  
  arma::mat lambda_ones(theta_mat.n_rows, theta_mat.n_cols);
  lambda_ones.fill(1);
  
  int log_counter = 0;

  arma::mat mu = mnre_mu_x_oop(fixed_effects, random_effects, 
                               mnre_beta_fixed, mnre_beta_random);
  
  arma::mat dy(n_data, k_class);
  
  arma::sp_mat sp_diag_mat = speye(n_dim_random * k_class, n_dim_random * k_class);
  int idx1, idx2;
  
  arma::sp_mat ftwf = fill_mtwm_x_oop(fixed_effects, fixed_effects, mu);
  arma::sp_mat ftwr = fill_mtwm_x_oop(fixed_effects, random_effects, mu);
  arma::sp_mat rtwr = fill_mtwm_x_oop(random_effects, random_effects, mu);
  
  if (DEBUG) {
    Rcpp::Rcout << " mnre step " <<
      " ndimfixed " << n_dim_fixed << " ndimrandom " << n_dim_random <<
        " ftwf " << ftwf.n_rows << " " << ftwf.n_cols <<
          " ftwr " << ftwr.n_rows << " " << ftwr.n_cols <<
            " rtwr " << rtwr.n_rows << " " << rtwr.n_cols <<
              std::endl;
  }
  
  // TODO: double check that the sp_diag_mat dimensions are correct, dont just fudge it like this
  //rtwr = rtwr + sp_diag_mat;
  rtwr = rtwr + speye(rtwr.n_rows, rtwr.n_cols);
  
  arma::sp_mat lhs_mat = arma::join_vert(
    arma::join_horiz(ftwf, ftwr),
    arma::join_horiz(ftwr.t(), rtwr)
  );
  
  if (DEBUG) {
    Rcpp::Rcout << " mnre step " <<
      " mu " << mu.n_rows << " " << mu.n_cols <<
        " dy " << dy.n_rows << " " << dy.n_cols << std::endl;
  }
  
  int iy;
  for (int i=0; i < y.n_rows; i++) {
    iy = (int)(y(i));
    for (int k=0; k < k_class; k++) {
      if (iy == k+1) {
        dy(i, k) = (1 - mu(i, k));
      } else {
        dy(i, k) = (0 - mu(i, k));
      }
    }
  }
  
  arma::mat rhs_fe = arma::trans(fixed_effects.submat(0, 0, n_data-1, n_dim_fixed-1)) * dy;
  arma::mat rhs_re = arma::trans(random_effects.submat(0, 0, n_data-1, n_dim_random-1)) * dy;
  
  if (DEBUG) {
    Rcpp::Rcout <<
      " n_dim_random " << n_dim_random <<
        " rhs_re " << rhs_re.n_rows << " " << rhs_re.n_cols <<
          std::endl;
    
    Rcpp::Rcout <<
      " n_dim_fixed " << n_dim_fixed <<
        " rhs_fe " << rhs_fe.n_rows << " " << rhs_fe.n_cols <<
          std::endl;
  }
  
  // for the penalty term. Note this assumes spherical random effects
  for (int d1=0; d1 < n_dim_random; d1++) {
    for (int k1=0; k1 < k_class; k1++) {
      rhs_re(d1, k1) -= mnre_beta_random(d1, k1);
    }
  }
  
  
  arma::mat rhs_mat = arma::join_vert(arma::vectorise(rhs_fe),
                                      arma::vectorise(rhs_re));
  
  double grad_check = arma::norm(arma::vectorise(rhs_mat % rhs_mat));
  
  if (DEBUG) {
    Rcpp::Rcout << " mnre step " <<
      " rhs_mat " << rhs_mat.n_rows << " " << rhs_mat.n_cols <<
        " lhs_mat " << lhs_mat.n_rows << " " << lhs_mat.n_cols <<
          std::endl;
  }
  
  //  SparseQR<SparseMatrix <double>, COLAMDOrdering<int> > solver;
  //  SimplicialLDLT<SparseMatrix <double> > solver;
  
  SparseMatrix<double> rx = Rcpp::as<SparseMatrix <double> >(wrap(lhs_mat));
  MatrixXd ry = Rcpp::as<MatrixXd>(wrap(rhs_mat));
  
  if (DEBUG) {
    Rcpp:Rcout << "solve the linear system..." << std::endl;
  }
  
  // use analyzePattern so the subsequent solve steps are quicker
  // note this means the pattern of zeros in the sparse matrix cannot change. 
  // I think this will be the case if we only update the random effects covariance matrix (theta_mat)
  // but need to be aware of this
    mnre_solver.compute(rx);
  
  // if (!MATRIX_ANALYZED) {
  //   set_solver_analyze(rx);
  // }
  // 
  // mnre_solver.factorize(rx);
  // 
  MatrixXd vsol = mnre_solver.solve(ry);
  
  //arma::vec vsol = spsolve(lhs_mat, rhs_mat, "lapack");
  
  arma::mat db = arma::reshape(
    as<arma::mat>(wrap(vsol.block(0, 0, Dfixed_times_K, 1))),
    n_dim_fixed, k_class
  );
  
  arma::mat du = arma::reshape(
    as<arma::mat>(wrap(vsol.block(Dfixed_times_K, 0, Drandom_times_K, 1))),
    n_dim_random, k_class
  );
  
  double loglk_old = mnre_lk_oop(mnre_beta_fixed, mnre_beta_random, y);
  double loglk = mnre_lk_oop(mnre_beta_fixed + db, mnre_beta_random + du, y);
  
  
  if (DEBUG) {
    Rcpp::Rcout << " mnre step " <<
      " beta_fixed " << mnre_beta_fixed.n_rows << " " << mnre_beta_fixed.n_cols <<
        " beta_random " << mnre_beta_random.n_rows << " " << mnre_beta_random.n_cols <<
          " db " << db.n_rows << " " << db.n_cols <<
            " du " << du.n_rows << " " << du.n_cols <<
              std::endl;
  }
  
  arma::mat beta_mat_old = arma::join_vert(mnre_beta_fixed, mnre_beta_random);
  arma::mat beta_mat = arma::join_vert(mnre_beta_fixed+db, mnre_beta_random+du);
  Rcpp::List ans;
  ans["loglk"] = loglk;
  ans["loglk_old"] = loglk_old;
  ans["beta_diff"] = db;
  ans["beta_mat_old"] = beta_mat_old;
  ans["beta_mat"] = beta_mat;
  ans["beta_fixed_old"] = mnre_beta_fixed;
  ans["beta_fixed"] = mnre_beta_fixed + db;
  ans["beta_fixed_diff"] = db;
  ans["beta_random_old"] = mnre_beta_random;
  ans["beta_random"] = mnre_beta_random + du;
  ans["beta_random_diff"] = du;
  ans["grad_check"] = grad_check;
  
  return ans;
}

void MNRE::set_solver_analyze(SparseMatrix<double>& rx) {
  mnre_solver.analyzePattern(rx);
  MATRIX_ANALYZED = true; 
}

int MNRE::mnre_dim_and_class_to_index(int i_dim, int i_class, int n_dim) {
  return i_class * n_dim + i_dim;
}

arma::sp_mat MNRE::mnre_make_covar_oop(double off_diagonal = 0.0) {
  
  int nr = Lind.n_rows;
  int nc = theta_mat.n_cols;
  arma::sp_mat covar_mat;
  
  int max_idx = nr * nc * nc;
  arma::umat covar_mat_idx(2, max_idx);
  arma::vec covar_mat_v(max_idx);
  int idx1, idx2, lidx;
  double theta1, theta2;
  int counter = 0;
  double covar_val;
  
  for (int ir=0; ir < nr; ir++) {
    lidx = Lind(ir)-1;
    for (int k1=0; k1 < nc; k1++) {
      theta1 = theta_mat(lidx, k1);
      idx1 = mnre_dim_and_class_to_index(ir, k1, nr);
      for (int k2=0; k2 < nc; k2++) {
        theta2 = theta_mat(lidx, k2);
        idx2 = mnre_dim_and_class_to_index(ir, k2, nr);
        
        covar_val = theta1 * theta2;
        if (k1 != k2) {
          covar_val *= off_diagonal;
        }
        
        covar_mat_idx(0, counter) = idx1;
        covar_mat_idx(1, counter) = idx2;
        covar_mat_v(counter) = covar_val;
        counter += 1;
        
      }
    }
  }
  
  covar_mat = arma::sp_mat(covar_mat_idx, covar_mat_v);
  return covar_mat;
}

double MNRE::mnre_lk_oop(const arma::mat& beta_fixed, 
                         const arma::mat& beta_random, 
                         const arma::vec& y) {
  //return 1.0;

  return mnre_lk_glm_oop(beta_fixed, beta_random, y) + mnre_lk_penalty_oop(beta_random, theta_mat);
}

double MNRE::mnre_lk_penalty_oop(const arma::mat& beta_random,
                       const arma::mat& theta_norm) {
  // note that when we work in spherical random effects the theta_norm term is identically 1
  // I'm leaving it here for now to potentially refactor later on and use this as a generic penalty
  // function.
  //return 1.0;
  
  int k_class = theta_norm.n_cols;
  double tmp;
  double lk_penalty = 0.0;
  int ind;

  for (int k1 = 0; k1 < k_class; k1++) {
    tmp = 0.0;
    for (int d1 = 0; d1 < beta_random.n_rows; d1++) { // dont penalize the intercept
      ind = Lind(d1)-1;
      // note we multiply by 1, not theta_mat, because it is assuemd we are in spherical coordinates.
      //      tmp += beta_random(d1, k1) * beta_random(d1, k1) * theta_norm(ind, k1);
      tmp += beta_random(d1, k1) * beta_random(d1, k1);
    }
    lk_penalty += tmp;
  }

  return lk_penalty;
  
}


double MNRE::mnre_lk_glm_oop(const arma::mat& beta_fixed,
                             const arma::mat& beta_random,
                             const arma::vec& y) {
  
  int n_data  = fixed_effects.n_rows;
  int n_dim_fixed   = fixed_effects.n_cols;
  int n_dim_random  = random_effects.n_cols;
  int k_class =  beta_fixed.n_cols;
  
  arma::mat mu = mnre_mu_x_oop(fixed_effects, random_effects,
                           beta_fixed, beta_random);
  
  double log_lk = 0.0, lk_term;
  int iy;
  
  for (int i=0; i < n_data; i++) {
    iy = (int) y(i);
    if (iy == 0) {
      lk_term = std::log(1 - arma::sum(mu.row(i)));
    } else {
      lk_term = std::log(mu(i,iy-1));
    }
    log_lk = log_lk + lk_term;
  }
  
  return -2 * log_lk;
}


//' @export
// [[Rcpp::export]]
RCPP_MODULE(mnre_mod) {
  using namespace Rcpp;
  class_<MNRE>("MNRE")
    .constructor()
    .constructor<int>()
    .constructor<arma::sp_mat, arma::sp_mat>()
    .constructor<arma::sp_mat, arma::sp_mat, int>()
  
  .method("mnre_echo", &MNRE::mnre_echo)

  // setters
  .method("set_k_class", &MNRE::set_k_class)
  .method("set_lind", &MNRE::set_lind)
  .method("set_beta", &MNRE::set_beta)
  .method("set_theta", &MNRE::set_theta)
  .method("set_y", &MNRE::set_y)
  .method("set_dims", &MNRE::set_dims)
  
  // getters
.method("get_k_class", &MNRE::get_k_class)
.method("get_lind", &MNRE::get_lind)
.method("get_beta", &MNRE::get_beta)
.method("get_theta", &MNRE::get_theta)
.method("get_y", &MNRE::get_y)
  
  .method("get_matrix_analyze_status", &MNRE::get_matrix_analyze_status)

  // fit methods
.method("mnre_oop_fit", &MNRE::mnre_oop_fit)
.method("mnre_lk_glm_oop", &MNRE::mnre_lk_glm_oop)
.method("mnre_lk_penalty_oop", &MNRE::mnre_lk_penalty_oop)
.method("mnre_lk_oop", &MNRE::mnre_lk_oop)
.method("mnre_step_oop", &MNRE::mnre_step_oop)
  
  // checks
.method("mnre_make_covar_oop", &MNRE::mnre_make_covar_oop)
.method("mnre_mu_x_oop", &MNRE::mnre_mu_x_oop)
.method("mnre_left_covar_factor_oop", &MNRE::mnre_left_covar_factor_oop)
    ;
}
