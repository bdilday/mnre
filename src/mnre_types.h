
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
public:
  int K_class;
  arma::sp_mat fixed_effects;
  arma::sp_mat random_effects;
  arma::mat y;
  arma::mat theta_mat;
  arma::uvec Lind;
  arma::mat beta_fixed;
  arma::mat beta_random;
  arma::mat fungible_theta;
  int verbose;
  
  MNRE(const arma::sp_mat fixed_effects,
       const arma::sp_mat random_effects,
       const arma::vec& y,
       const arma::mat& theta_norm,
       const arma::uvec& Lind,
       arma::mat beta_fixed,
       arma::mat beta_random,
       int verbose);
  
  MNRE();
  
  void set_theta(arma::mat &theta_norm);
  void mnre_echo(NumericVector &x);
  
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


