// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppEigen)]]

#include <RcppArmadillo.h>
#include <RcppEigen.h>

#ifndef mnre_H
#define mnre_H


using Eigen::ArrayXd;
using Eigen::LLT;
using Eigen::MatrixXd;
using Eigen::VectorXd;

using Eigen::SparseMatrix;
using Eigen::SparseQR;

using namespace Rcpp;
using namespace Eigen;

arma::sp_mat fill_mtwm_x(const arma::sp_mat& x1, const arma::sp_mat& x2,
                         const arma::mat& mu);

Rcpp::List mnre_fit_sparse(const arma::sp_mat& fixed_effects,
                               const arma::sp_mat& random_effects,
                               const arma::vec& y,
                               const arma::mat& theta_norm,
                               const arma::uvec& Lind,
                               arma::mat beta_fixed,
                               arma::mat beta_random,
                               int verbose);


double mnre_lk_glm(const arma::sp_mat& fixed_effects,
                       const arma::sp_mat& random_effects,
                       const arma::mat& beta_fixed,
                       const arma::mat& beta_random,
                       const arma::vec& y,
                       const arma::umat& Lind);

double mnre_lk(const arma::sp_mat& fixed_effects,
                   const arma::sp_mat& random_effects,
                   const arma::mat& beta_fixed,
                   const arma::mat& beta_random,
                   const arma::vec &y,
                   const arma::mat &lambda_norm,
                   const arma::umat& Lind);

arma::mat mnre_mu_x(const arma::sp_mat &fe_x,
                        const arma::sp_mat &re_x,
                        const arma::mat &beta_fixed,
                        const arma::mat &beta_random);

arma::mat mnre_mu(const arma::mat &fixed_effects,
                      const arma::sp_mat &random_effects,
                      const arma::mat &beta_fixed,
                      const arma::mat &beta_random);

arma::sp_mat mnre_expand_matrix(const arma::sp_mat& x1, int k_class, int directions);

arma::sp_mat mnre_left_covar_factor(arma::sp_mat& x1);

arma::sp_mat mnre_make_covar(const arma::mat& theta_mat,
                             const arma::umat& Lind,
                             double off_diagonal);


Rcpp::List mnre_step_sparse(const arma::sp_mat &fixed_effects,
                            const arma::sp_mat &random_effects,
                            const arma::vec &y,
                            const arma::mat &beta_fixed,
                            const arma::mat &beta_random,
                            const arma::mat &lambda_norm,
                            const arma::uvec &Lind);

#endif