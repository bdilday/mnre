// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "mnre_types.h"
#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// mnre_dim_and_class_to_index
int mnre_dim_and_class_to_index(int i_dim, int i_class, int n_dim);
RcppExport SEXP _mnre_mnre_dim_and_class_to_index(SEXP i_dimSEXP, SEXP i_classSEXP, SEXP n_dimSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type i_dim(i_dimSEXP);
    Rcpp::traits::input_parameter< int >::type i_class(i_classSEXP);
    Rcpp::traits::input_parameter< int >::type n_dim(n_dimSEXP);
    rcpp_result_gen = Rcpp::wrap(mnre_dim_and_class_to_index(i_dim, i_class, n_dim));
    return rcpp_result_gen;
END_RCPP
}
// mnre_make_covar
arma::sp_mat mnre_make_covar(const arma::mat& theta_mat, const arma::umat& Lind, double off_diagonal);
RcppExport SEXP _mnre_mnre_make_covar(SEXP theta_matSEXP, SEXP LindSEXP, SEXP off_diagonalSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type theta_mat(theta_matSEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type Lind(LindSEXP);
    Rcpp::traits::input_parameter< double >::type off_diagonal(off_diagonalSEXP);
    rcpp_result_gen = Rcpp::wrap(mnre_make_covar(theta_mat, Lind, off_diagonal));
    return rcpp_result_gen;
END_RCPP
}
// mnre_expand_matrix
arma::sp_mat mnre_expand_matrix(const arma::sp_mat& x1, int k_class, int direction);
RcppExport SEXP _mnre_mnre_expand_matrix(SEXP x1SEXP, SEXP k_classSEXP, SEXP directionSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type x1(x1SEXP);
    Rcpp::traits::input_parameter< int >::type k_class(k_classSEXP);
    Rcpp::traits::input_parameter< int >::type direction(directionSEXP);
    rcpp_result_gen = Rcpp::wrap(mnre_expand_matrix(x1, k_class, direction));
    return rcpp_result_gen;
END_RCPP
}
// mnre_left_covar_factor
arma::sp_mat mnre_left_covar_factor(arma::sp_mat& x1);
RcppExport SEXP _mnre_mnre_left_covar_factor(SEXP x1SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::sp_mat& >::type x1(x1SEXP);
    rcpp_result_gen = Rcpp::wrap(mnre_left_covar_factor(x1));
    return rcpp_result_gen;
END_RCPP
}
// mnre_fit_sparse
Rcpp::List mnre_fit_sparse(const arma::sp_mat& fixed_effects, const arma::sp_mat& random_effects, const arma::vec& y, const arma::mat& theta_mat, const arma::uvec& Lind, arma::mat beta_fixed, arma::mat beta_random, int verbose);
RcppExport SEXP _mnre_mnre_fit_sparse(SEXP fixed_effectsSEXP, SEXP random_effectsSEXP, SEXP ySEXP, SEXP theta_matSEXP, SEXP LindSEXP, SEXP beta_fixedSEXP, SEXP beta_randomSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type fixed_effects(fixed_effectsSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type random_effects(random_effectsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta_mat(theta_matSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type Lind(LindSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type beta_fixed(beta_fixedSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type beta_random(beta_randomSEXP);
    Rcpp::traits::input_parameter< int >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(mnre_fit_sparse(fixed_effects, random_effects, y, theta_mat, Lind, beta_fixed, beta_random, verbose));
    return rcpp_result_gen;
END_RCPP
}
// mnre_lk_penalty
double mnre_lk_penalty(const arma::mat& beta_random, const arma::mat& theta_norm, const arma::umat& Lind);
RcppExport SEXP _mnre_mnre_lk_penalty(SEXP beta_randomSEXP, SEXP theta_normSEXP, SEXP LindSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_random(beta_randomSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta_norm(theta_normSEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type Lind(LindSEXP);
    rcpp_result_gen = Rcpp::wrap(mnre_lk_penalty(beta_random, theta_norm, Lind));
    return rcpp_result_gen;
END_RCPP
}
// mnre_lk_glm
double mnre_lk_glm(const arma::sp_mat& fixed_effects, const arma::sp_mat& random_effects, const arma::mat& beta_fixed, const arma::mat& beta_random, const arma::vec& y, const arma::umat& Lind);
RcppExport SEXP _mnre_mnre_lk_glm(SEXP fixed_effectsSEXP, SEXP random_effectsSEXP, SEXP beta_fixedSEXP, SEXP beta_randomSEXP, SEXP ySEXP, SEXP LindSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type fixed_effects(fixed_effectsSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type random_effects(random_effectsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_fixed(beta_fixedSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_random(beta_randomSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type Lind(LindSEXP);
    rcpp_result_gen = Rcpp::wrap(mnre_lk_glm(fixed_effects, random_effects, beta_fixed, beta_random, y, Lind));
    return rcpp_result_gen;
END_RCPP
}
// mnre_lk
double mnre_lk(const arma::sp_mat& fixed_effects, const arma::sp_mat& random_effects, const arma::mat& beta_fixed, const arma::mat& beta_random, const arma::vec& y, const arma::mat& theta_norm, const arma::umat& Lind);
RcppExport SEXP _mnre_mnre_lk(SEXP fixed_effectsSEXP, SEXP random_effectsSEXP, SEXP beta_fixedSEXP, SEXP beta_randomSEXP, SEXP ySEXP, SEXP theta_normSEXP, SEXP LindSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type fixed_effects(fixed_effectsSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type random_effects(random_effectsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_fixed(beta_fixedSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_random(beta_randomSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta_norm(theta_normSEXP);
    Rcpp::traits::input_parameter< const arma::umat& >::type Lind(LindSEXP);
    rcpp_result_gen = Rcpp::wrap(mnre_lk(fixed_effects, random_effects, beta_fixed, beta_random, y, theta_norm, Lind));
    return rcpp_result_gen;
END_RCPP
}
// mnre_mu
arma::mat mnre_mu(const arma::mat& fixed_effects, const arma::sp_mat& random_effects, const arma::mat& beta_fixed, const arma::mat& beta_random);
RcppExport SEXP _mnre_mnre_mu(SEXP fixed_effectsSEXP, SEXP random_effectsSEXP, SEXP beta_fixedSEXP, SEXP beta_randomSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type fixed_effects(fixed_effectsSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type random_effects(random_effectsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_fixed(beta_fixedSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_random(beta_randomSEXP);
    rcpp_result_gen = Rcpp::wrap(mnre_mu(fixed_effects, random_effects, beta_fixed, beta_random));
    return rcpp_result_gen;
END_RCPP
}
// mnre_mu_x
arma::mat mnre_mu_x(const arma::sp_mat& fe_x, const arma::sp_mat& re_x, const arma::mat& beta_fixed, const arma::mat& beta_random);
RcppExport SEXP _mnre_mnre_mu_x(SEXP fe_xSEXP, SEXP re_xSEXP, SEXP beta_fixedSEXP, SEXP beta_randomSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type fe_x(fe_xSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type re_x(re_xSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_fixed(beta_fixedSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_random(beta_randomSEXP);
    rcpp_result_gen = Rcpp::wrap(mnre_mu_x(fe_x, re_x, beta_fixed, beta_random));
    return rcpp_result_gen;
END_RCPP
}
// mnre_step_sparse
Rcpp::List mnre_step_sparse(const arma::sp_mat& fixed_effects, const arma::sp_mat& random_effects, const arma::vec& y, const arma::mat& beta_fixed, const arma::mat& beta_random, const arma::mat& lambda_norm, const arma::uvec& Lind);
RcppExport SEXP _mnre_mnre_step_sparse(SEXP fixed_effectsSEXP, SEXP random_effectsSEXP, SEXP ySEXP, SEXP beta_fixedSEXP, SEXP beta_randomSEXP, SEXP lambda_normSEXP, SEXP LindSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type fixed_effects(fixed_effectsSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type random_effects(random_effectsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_fixed(beta_fixedSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_random(beta_randomSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type lambda_norm(lambda_normSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type Lind(LindSEXP);
    rcpp_result_gen = Rcpp::wrap(mnre_step_sparse(fixed_effects, random_effects, y, beta_fixed, beta_random, lambda_norm, Lind));
    return rcpp_result_gen;
END_RCPP
}
// fill_mtwm_x
arma::sp_mat fill_mtwm_x(const arma::sp_mat& x1, const arma::sp_mat& x2, const arma::mat& mu);
RcppExport SEXP _mnre_fill_mtwm_x(SEXP x1SEXP, SEXP x2SEXP, SEXP muSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type x1(x1SEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type x2(x2SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type mu(muSEXP);
    rcpp_result_gen = Rcpp::wrap(fill_mtwm_x(x1, x2, mu));
    return rcpp_result_gen;
END_RCPP
}
// mnre_create_empty
SEXP mnre_create_empty();
RcppExport SEXP _mnre_mnre_create_empty() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(mnre_create_empty());
    return rcpp_result_gen;
END_RCPP
}
// mnre_create
SEXP mnre_create(const arma::sp_mat fixed_effects, const arma::sp_mat random_effects, const arma::vec& y, const arma::mat& theta_norm, const arma::uvec& Lind, arma::mat beta_fixed, arma::mat beta_random, int verbose);
RcppExport SEXP _mnre_mnre_create(SEXP fixed_effectsSEXP, SEXP random_effectsSEXP, SEXP ySEXP, SEXP theta_normSEXP, SEXP LindSEXP, SEXP beta_fixedSEXP, SEXP beta_randomSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::sp_mat >::type fixed_effects(fixed_effectsSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat >::type random_effects(random_effectsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta_norm(theta_normSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type Lind(LindSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type beta_fixed(beta_fixedSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type beta_random(beta_randomSEXP);
    Rcpp::traits::input_parameter< int >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(mnre_create(fixed_effects, random_effects, y, theta_norm, Lind, beta_fixed, beta_random, verbose));
    return rcpp_result_gen;
END_RCPP
}

RcppExport SEXP _rcpp_module_boot_mnre_mod();

static const R_CallMethodDef CallEntries[] = {
    {"_mnre_mnre_dim_and_class_to_index", (DL_FUNC) &_mnre_mnre_dim_and_class_to_index, 3},
    {"_mnre_mnre_make_covar", (DL_FUNC) &_mnre_mnre_make_covar, 3},
    {"_mnre_mnre_expand_matrix", (DL_FUNC) &_mnre_mnre_expand_matrix, 3},
    {"_mnre_mnre_left_covar_factor", (DL_FUNC) &_mnre_mnre_left_covar_factor, 1},
    {"_mnre_mnre_fit_sparse", (DL_FUNC) &_mnre_mnre_fit_sparse, 8},
    {"_mnre_mnre_lk_penalty", (DL_FUNC) &_mnre_mnre_lk_penalty, 3},
    {"_mnre_mnre_lk_glm", (DL_FUNC) &_mnre_mnre_lk_glm, 6},
    {"_mnre_mnre_lk", (DL_FUNC) &_mnre_mnre_lk, 7},
    {"_mnre_mnre_mu", (DL_FUNC) &_mnre_mnre_mu, 4},
    {"_mnre_mnre_mu_x", (DL_FUNC) &_mnre_mnre_mu_x, 4},
    {"_mnre_mnre_step_sparse", (DL_FUNC) &_mnre_mnre_step_sparse, 7},
    {"_mnre_fill_mtwm_x", (DL_FUNC) &_mnre_fill_mtwm_x, 3},
    {"_mnre_mnre_create_empty", (DL_FUNC) &_mnre_mnre_create_empty, 0},
    {"_mnre_mnre_create", (DL_FUNC) &_mnre_mnre_create, 8},
    {"_rcpp_module_boot_mnre_mod", (DL_FUNC) &_rcpp_module_boot_mnre_mod, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_mnre(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
