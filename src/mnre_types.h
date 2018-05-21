
#ifndef mnre_types_H
#define mnre_types_H

#include <gperftools/profiler.h>
#include <RcppArmadillo.h>
#include <RcppEigen.h>

using Eigen::ArrayXd;
using Eigen::LLT;
using Eigen::MatrixXd;
using Eigen::VectorXd;

using Eigen::SparseMatrix;
using Eigen::SparseQR;

using namespace Rcpp;
using namespace Eigen;
using namespace arma;

class MNRE {
  
private:
  int k_class;
  arma::sp_mat fixed_effects;
  arma::sp_mat random_effects;
  arma::vec y;
  arma::mat theta_mat;
  arma::uvec Lind;
  arma::mat mnre_beta_fixed;
  arma::mat mnre_beta_random;
  arma::mat mnre_mu;
  arma::sp_mat mnre_covar_mat;
  arma::sp_mat mnre_ZLam;
  arma::sp_mat mnre_fe_x;
  arma::sp_mat mnre_re_x;
  SimplicialLLT<SparseMatrix <double> > mnre_solver;
  
  int n_data;
  int n_dim_fixed;
  int n_dim_random;
  int n_dim; 
  int D_times_K;
  int Dfixed_times_K;
  int Drandom_times_K;
  
  
public:
  int verbose = 0;
  bool MATRIX_ANALYZED = false;
  
  
  MNRE(const arma::sp_mat& fixed_effects,
       const arma::sp_mat& random_effects
  );
  
  MNRE(const arma::sp_mat& fixed_effects,
       const arma::sp_mat& random_effects,
       int verbose
  );
  
  MNRE();
  
  MNRE(int x);
  
  void set_k_class(Rcpp::IntegerVector k_class_) {
    k_class = k_class_(0);
  };
  
  int get_k_class() {
    return k_class;
  };
  
  void set_lind(const arma::uvec& Lind_) {
    Lind = Lind_;
  };
  
  arma::uvec get_lind() {
    return Lind;
  };
  
  void set_beta(const arma::mat &bf, const arma::mat &br) {
    mnre_beta_fixed = bf;
    mnre_beta_random = br;
  };
  
  Rcpp::List get_beta() {
    Rcpp::List ans;  
    ans["beta_fixed"] = mnre_beta_fixed;
    ans["beta_random"] = mnre_beta_random;
    return ans;
  };
  
  void set_theta(arma::mat &m) {
    theta_mat = m;
  };
  
  arma::mat get_theta() {
    return theta_mat;
  };
  
  void set_y(const arma::vec& y_) {
    y = y_;
    k_class = arma::max(y);
  }
  
  arma::vec get_y() {
    return y;
  }
  
  arma::sp_mat get_covar_mat() {
    return mnre_covar_mat;  
  }
  
  arma::mat get_mu() {
    return mnre_mu;
  };
  
  arma::sp_mat get_ZLam() {
    return mnre_ZLam;
  };
  
  arma::sp_mat mnre_oop_expand_matrix(const arma::sp_mat& x1, int k_class, int direction);
  
  void set_solver_analyze(SparseMatrix<double>& rx);
  
  arma::sp_mat mnre_left_covar_factor_oop(arma::sp_mat& x1);
  int mnre_dim_and_class_to_index(int i_dim, int i_class, int n_dim);
  
  void mnre_echo(NumericVector &x);
  
  void set_dims();
  
  Rcpp::List mnre_oop_fit();
  Rcpp::List mnre_step_oop();
  
  arma::sp_mat mnre_make_covar_oop(double off_diagonal);
  
  bool get_matrix_analyze_status() {
    return MATRIX_ANALYZED;
  }
  
  arma::sp_mat fill_mtwm_x_oop(const arma::sp_mat& x1, const arma::sp_mat& x2,
                           const arma::mat& mu);
  
  double mnre_lk_oop(const arma::sp_mat& fe_, 
                     const arma::sp_mat& re_,
                     const arma::mat& beta_fixed,
                     const arma::mat& beta_random, 
                     const arma::vec& y);
  
  double mnre_lk_penalty_oop(const arma::mat& beta_random,
                             const arma::mat& theta_norm);
  
  double mnre_lk_glm_oop(const arma::sp_mat& fe_, 
                         const arma::sp_mat& re_,
                         const arma::mat& beta_fixed,
                         const arma::mat& beta_random,
                         const arma::vec& y);
  
  arma::mat mnre_mu_oop(const arma::mat &fixed_effects,
                        const arma::sp_mat &random_effects,
                        const arma::mat &beta_fixed,
                        const arma::mat &beta_random);
  
  arma::mat mnre_mu_x_oop(const arma::sp_mat &fe_x,
                          const arma::sp_mat &re_x,
                          const arma::mat &beta_fixed,
                          const arma::mat &beta_random);
    
    
};


#endif


