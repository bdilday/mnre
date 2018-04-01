
#' @export
mnre_cpp = setRefClass("mnre_cpp",
                       fields = 
                         list(Ptr = "externalptr",
                              fixed_effects = "dgCMatrix",
                              random_effects   = "dgCMatrix",
                              theta_mat = "numeric"
                         ),
                       
                       methods = list(
                         initialize = function() {},
                         setTheta = function() {}
                       )
)