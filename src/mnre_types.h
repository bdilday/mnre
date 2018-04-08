
#ifndef mnre_types_H
#define mnre_types_H

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
  arma::mat beta_fixed;
  arma::mat beta_random;
  arma::sp_mat covar_mat;
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
  
  
  MNRE(const arma::sp_mat fixed_effects,
       const arma::sp_mat random_effects
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
    beta_fixed = bf;
    beta_random = br;
  };
  
  Rcpp::List get_beta() {
    Rcpp::List ans;  
    ans["beta_fixed"] = beta_fixed;
    ans["beta_random"] = beta_random;
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
  
  arma::sp_mat mnre_oop_expand_matrix(const arma::sp_mat& x1, int k_class, int direction);
  
  void set_solver_analyze(SparseMatrix<double>& rx);
  
  arma::sp_mat mnre_left_covar_factor(arma::sp_mat& x1);
  int mnre_dim_and_class_to_index(int i_dim, int i_class, int n_dim);
  
  void mnre_echo(NumericVector &x);
  
  void set_dims();
  
  Rcpp::List mnre_oop_fit();
  Rcpp::List mnre_step_oop();
  
  arma::sp_mat mnre_make_covar(double off_diagonal);
  
  bool get_matrix_analyze_status() {
    return MATRIX_ANALYZED;
  }
  
  
  
};



#endif


