/* 
 * Authors: Dong Shufeng
 * 2011-11-21
 */
#include <jni.h>
#include "IpTNLP.hpp"
#include "IpIpoptApplication.hpp"
#include "org_coinor_Ipopt.h"
#include "IpSparseSymLinearSolverInterface.hpp"
#include "IpPardisoSolverInterface.hpp"
#include "IpMumpsSolverInterface.hpp"
#include "IpCustomSolverInterface.hpp"
#include "IpMa27TSolverInterface.hpp"
#include "IpTSymScalingMethod.hpp"
#include "IpSlackBasedTSymScalingMethod.hpp"
#include "IpTSymLinearSolver.hpp"
#include "IpStdAugSystemSolver.hpp"
#include "IpAlgBuilder.hpp"
#include "IpInexactAlgBuilder.hpp"
#include "IpTNLPAdapter.hpp"


using namespace std;
using namespace Ipopt;

/**
 * Main structure for Ipopt JNI implementation.
 * 
 * All functions will receive a pointer to this structure as
 * an integer argument (the address in memory of the structure).
 */
class Jipopt : public TNLP {
public:
  /**  constructor */
  Jipopt(JNIEnv *env, jobject solver, jint n, jint m, jint nele_jac, jint nele_hess, jint index_style);

  /** default destructor */
  virtual ~Jipopt();

  /**@name Overloaded from TNLP */
  //@{
  /** Method to return some info about the nlp */
  virtual bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                            Index& nnz_h_lag, IndexStyleEnum& index_style);

  /** Method to return the bounds for my problem */
  virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u,
                               Index m, Number* g_l, Number* g_u);

  /** Method to return the starting point for the algorithm */
  virtual bool get_starting_point(Index n, bool init_x, Number* x,
                                  bool init_z, Number* z_L, Number* z_U,
                                  Index m, bool init_lambda,
                                  Number* lambda);

  /** Method to return the objective value */
  virtual bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value);

  /** Method to return the gradient of the objective */
  virtual bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f);

  /** Method to return the constraint residuals */
  virtual bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g);

  /** Method to return:
   *   1) The structure of the jacobian (if "values" is NULL)
   *   2) The values of the jacobian (if "values" is not NULL)
   */
  virtual bool eval_jac_g(Index n, const Number* x, bool new_x,
                          Index m, Index nele_jac, Index* iRow, Index *jCol,
                          Number* values);

  /** Method to return:
   *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
   *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
   */
  virtual bool eval_h(Index n, const Number* x, bool new_x,
                      Number obj_factor, Index m, const Number* lambda,
                      bool new_lambda, Index nele_hess, Index* iRow,
                      Index* jCol, Number* values);

  //@}

  /** @name Solution Methods */
  //@{
  /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
  virtual void finalize_solution(SolverReturn status,
                                 Index n, const Number* x, const Number* z_L, const Number* z_U,
                                 Index m, const Number* g, const Number* lambda,
                                 Number obj_value,
				 const IpoptData* ip_data,
				 IpoptCalculatedQuantities* ip_cq);
  //@}



  /** overload this method to return scaling parameters. This is
     *  only called if the options are set to retrieve user scaling.
     *  There, use_x_scaling (or use_g_scaling) should get set to true
     *  only if the variables (or constraints) are to be scaled.  This
     *  method should return true only if the scaling parameters could
     *  be provided.
     */
    virtual bool get_scaling_parameters(Number& obj_scaling,
                                        bool& use_x_scaling, Index n,
                                        Number* x_scaling,
                                        bool& use_g_scaling, Index m,
                                        Number* g_scaling);
   


  /** @name Methods for quasi-Newton approximation.  If the second
     *  derivatives are approximated by Ipopt, it is better to do this
     *  only in the space of nonlinear variables.  The following
     *  methods are call by Ipopt if the quasi-Newton approximation is
     *  selected.  If -1 is returned as number of nonlinear variables,
     *  Ipopt assumes that all variables are nonlinear.  Otherwise, it
     *  calls get_list_of_nonlinear_variables with an array into which
     *  the indices of the nonlinear variables should be written - the
     *  array has the lengths num_nonlin_vars, which is identical with
     *  the return value of get_number_of_nonlinear_variables().  It
     *  is assumed that the indices are counted starting with 1 in the
     *  FORTRAN_STYLE, and 0 for the C_STYLE. */
    //@{
    virtual Index get_number_of_nonlinear_variables();
    

    virtual bool get_list_of_nonlinear_variables(Index num_nonlin_vars,Index* pos_nonlin_vars);

public:
	// The JNI Environment
	JNIEnv *env;
	jobject solver;
	
	jint n;
	jint m;
	jint nele_jac;
	jint nele_hess;

	jint index_style;

	// some cached arguments
	jdoubleArray mult_gj;
	
	// the callback arguments
	jdoubleArray xj;
	jdoubleArray fj;
	jdoubleArray grad_fj;
	jdoubleArray gj;
	jdoubleArray jac_gj;
	jdoubleArray hessj;

	jdoubleArray mult_x_Lj;
	jdoubleArray mult_x_Uj;

	jboolean using_scaling_parameters;
	jboolean using_LBFGS;

  	// the callback methods	
	//jmethodID get_nlp_info_;
	jmethodID get_bounds_info_;
	jmethodID get_starting_point_;
	jmethodID eval_f_;
	jmethodID eval_grad_f_;
	jmethodID eval_g_;
	jmethodID eval_jac_g_;
	jmethodID eval_h_;

	jmethodID get_scaling_parameters_;
	jmethodID get_number_of_nonlinear_variables_;
	jmethodID get_list_of_nonlinear_variables_;
	

private:
  /**@name Methods to block default compiler methods.
   * The compiler automatically generates the following three methods.
   *  Since the default compiler implementation is generally not what
   *  you want (for all but the most simple classes), we usually 
   *  put the declarations of these methods in the private section
   *  and never implement them. This prevents the compiler from
   *  implementing an incorrect "default" behavior without us
   *  knowing. (See Scott Meyers book, "Effective C++")
   *  
   */
  //@{
  //  MyNLP();
  Jipopt(const Jipopt&);
  Jipopt& operator=(const Jipopt&);
  
  //@}
};

class JIpoptSolver {
public:
	SmartPtr<IpoptApplication> application; 
	Jipopt * problem;
};
