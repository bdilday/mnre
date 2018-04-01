// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppEigen)]]

#include <RcppArmadillo.h>
#include <RcppEigen.h>

#ifndef mnre_H_OOP
#define mnre_H_OOP

using Eigen::ArrayXd;
using Eigen::LLT;
using Eigen::MatrixXd;
using Eigen::VectorXd;

using Eigen::SparseMatrix;
using Eigen::SparseQR;

using namespace Rcpp;
using namespace Eigen;

class mnre {
public:
  int K_class;
  arma::sp_mat fixed_effects;
  arma::sp_mat random_effects;
  arma::mat y;
  arma::mat theta_mat;
  arma::uvec Lind;
  arma::mat beta_fixed;
  arma::mat beta_random;
  int verbose;
    
  mnre(const arma::sp_mat fixed_effects,
       const arma::sp_mat random_effects,
       const arma::vec& y,
       const arma::mat& theta_norm,
       const arma::uvec& Lind,
       arma::mat beta_fixed,
       arma::mat beta_random,
       int verbose);

  mnre();
  
  void set_theta(arma::mat theta_norm);
  
  Rcpp::List mnre_fit_sparse(const arma::sp_mat& fixed_effects,
                             const arma::sp_mat& random_effects,
                             const arma::vec& y,
                             const arma::mat& theta_norm,
                             const arma::uvec& Lind,
                             arma::mat beta_fixed,
                             arma::mat beta_random,
                             int verbose);
  

};



#endif