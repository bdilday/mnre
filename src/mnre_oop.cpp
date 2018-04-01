
#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include "mnre.h"
#include "mnre_oop.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppEigen)]]

using namespace arma;

#define DEBUG 0
#define EXPAND_ROW 0
#define EXPAND_COLUMN 1

//' @export
// [[Rcpp::export]]
mnre::mnre(const arma::sp_mat fixed_effects,
           const arma::sp_mat random_effects,
           const arma::vec& y,
           const arma::mat& theta_norm,
           const arma::uvec& Lind,
           arma::mat beta_fixed,
           arma::mat beta_random,
           int verbose): 
  fixed_effects (fixed_effects),
  random_effects (random_effects),
  y (y),
  theta_mat (theta_norm),
  Lind (Lind), 
  beta_fixed (beta_fixed) {
  Rcpp::Rcout << "I am mnre!" << std::endl;
};

//' @export
// [[Rcpp::export]]
mnre::mnre() {
  Rcpp::Rcout << "I am mnre!" << std::endl;
};

// //' @export
// // [[Rcpp::export]]
// SEXP mnre_create() {
//   mnre *m = new mnre();
//   return wrap(XPtr<mnre>(m, true));
// }

//' @export
// [[Rcpp::export]]
SEXP mnre_create(const arma::sp_mat fixed_effects,
                 const arma::sp_mat random_effects,
                 const arma::vec& y,
                 const arma::mat& theta_norm,
                 const arma::uvec& Lind,
                 arma::mat beta_fixed,
                 arma::mat beta_random,
                 int verbose) {
  mnre *m = new mnre(fixed_effects,
                     random_effects,
                     y,
                     theta_norm,
                     Lind,
                     beta_fixed,
                     beta_random,
                    verbose);
  return wrap(XPtr<mnre>(m, true));
}

Rcpp::List mnre::mnre_fit_sparse(const arma::sp_mat& fixed_effects,
                           const arma::sp_mat& random_effects,
                           const arma::vec& y,
                           const arma::mat& theta_mat,
                           const arma::uvec& Lind,
                           arma::mat beta_fixed,
                           arma::mat beta_random, 
                           int verbose=1) {
  arma::mat lambda_ones(theta_mat.n_rows, theta_mat.n_cols);
  lambda_ones.fill(1);
  
  int n_data  = fixed_effects.n_rows;
  int n_dim_fixed   = fixed_effects.n_cols;
  int n_dim_random  = random_effects.n_cols;
  int k_class = (int) y.max();
  int n_dim = n_dim_fixed + n_dim_random;
  
  int D_times_K = n_dim * k_class;
  
  // arma::mat beta_fixed = arma::mat(n_dim_fixed, k_class);
  // arma::mat beta_random = arma::mat(n_dim_random, k_class);
  // beta_fixed.fill(0);
  // beta_random.fill(0);
  
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
  
  arma::sp_mat covar_mat = mnre_make_covar(theta_mat, Lind, 0.0);
  arma::mat mu;
  arma::sp_mat left_factor = mnre_left_covar_factor(covar_mat);
  
  arma::sp_mat re_x = mnre_expand_matrix(random_effects, k_class, EXPAND_COLUMN);
  arma::sp_mat fe_x = mnre_expand_matrix(fixed_effects, k_class, EXPAND_COLUMN);
  
  if (DEBUG) {
    Rcpp::Rcout << " left_factor " << left_factor.n_rows << " " << left_factor.n_cols << std::endl;
    Rcpp::Rcout << " random_effects " << random_effects.n_rows << " " << random_effects.n_cols << std::endl;
    Rcpp::Rcout << " re_x " << re_x.n_rows << " " << re_x.n_cols << " " << re_x(1, 2) << std::endl;
  }
  
  arma::sp_mat ZLam = re_x * left_factor;
  
  if (DEBUG) {
    Rcpp::Rcout << " ZLam " << ZLam.n_rows << " " << ZLam.n_cols << " " << ZLam(1, 2) << std::endl;
  }
  
  arma::sp_mat rtwr;
  arma::sp_mat sp_diag_mat = speye(n_dim_random * k_class, n_dim_random * k_class);
  SimplicialLLT<SparseMatrix <double> > solver;
  SparseMatrix<double> rx;
  
  for (int istep=0; istep < max_step; istep++) {
    if (DEBUG) {
      Rcpp:Rcout << " istep " << istep << std::endl;
    }
    
    mu = mnre_mu_x(fe_x, ZLam, beta_fixed, beta_random);
    
    if (DEBUG) {
      Rcpp::Rcout << " call step... " << std::endl;
    }
    
    ll = mnre_step_sparse(fe_x, ZLam,
                          y,
                          beta_fixed, beta_random,
                          theta_mat, Lind);
    
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
          mnre_lk(fe_x, ZLam,
                  beta_fixed +
                    newton_step_factor * as<arma::mat>(ll["beta_fixed_diff"]),
                    beta_random +
                      newton_step_factor * as<arma::mat>(ll["beta_random_diff"]),
                      y, lambda_ones, Lind);
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
      rtwr = fill_mtwm_x(ZLam, ZLam, mu);
      rtwr = rtwr + sp_diag_mat;
      
      // TODO: either use the left factor or use rtwr
      rx = Rcpp::as<SparseMatrix <double> >(wrap(rtwr));
      solver.compute(rx);
      left_factor = Rcpp::as<arma::sp_mat>(wrap(solver.matrixL()));
      //loglk_det = arma::sum(std::log(left_factor.diag()));
      //loglk_det = 0;
      arma::log_det(loglk_det_val, loglk_det_sign, arma::mat(rtwr));
      loglk_det = loglk_det_val * loglk_det_sign;
      
      if (verbose >= 1) {
        Rcpp::Rcout << "half step cycle did not find a better fit" << std::endl;
      }
      
      ans["beta_fixed"] = beta_fixed;
      ans["beta_random"] = beta_random;
      ans["loglk"] = (double)(ll["loglk"]);
      ans["loglk_det"] = loglk_det;
      
      return ans;
    } else {
      beta_fixed = beta_fixed + newton_step_factor * as<arma::mat>(ll["beta_fixed_diff"]);
      beta_random = beta_random + newton_step_factor * as<arma::mat>(ll["beta_random_diff"]);
      newton_step_factor = 1.0;
    }
    
    if (std::abs(lk_diff_stat) <= tol && (double)ll["grad_check"] <= grad_tol) {
      rtwr = fill_mtwm_x(ZLam, ZLam, mu);
      rtwr = rtwr + sp_diag_mat;
      // TODO: either use the left factor or use rtwr
      rx = Rcpp::as<SparseMatrix <double> >(wrap(rtwr));
      solver.compute(rx);
      left_factor = Rcpp::as<arma::sp_mat>(wrap(solver.matrixL()));
      //loglk_det = arma::sum(std::log(left_factor.diag()));
      arma::log_det(loglk_det_val, loglk_det_sign, arma::mat(rtwr));
      loglk_det = loglk_det_val * loglk_det_sign;
      
      ans["beta_fixed"] = beta_fixed;
      ans["beta_random"] = beta_random;
      ans["loglk"] = (double)(ll["loglk"]);
      ans["loglk_det"] = loglk_det;
      return ans;
    }
    
  } // steps
  
} // end function
