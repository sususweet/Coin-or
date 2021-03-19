#include <jni.h>
#include "jipopt.hpp"
#include "org_coinor_Ipopt.h"
#include "org_coinor_Bonmin.h"

#include <iomanip>
#include "BonTMINLP.hpp"
#include "CoinPragma.hpp"
#include "CoinTime.hpp"
#include "CoinError.hpp"

#include "BonOsiTMINLPInterface.hpp"
#include "BonIpoptSolver.hpp"
#include "BonCbc.hpp"
#include "BonBonminSetup.hpp"

#include "BonOACutGenerator2.hpp"
#include "BonEcpCuts.hpp"
#include "BonOaNlpOptim.hpp"

using namespace std;
using namespace Ipopt;
using namespace Bonmin;

class JBonmin : public TMINLP {
public:
	/// Default constructor.
	JBonmin(JNIEnv *env, jobject solver, jint n, jint m);
	
  
	/// virtual destructor.
	virtual ~JBonmin(){delete ipoptProblem;}

	/** \name Overloaded functions specific to a TMINLP.*/
	//@{
	/** Pass the type of the variables (INTEGER, BINARY, CONTINUOUS) to the optimizer.
	\param n size of var_types (has to be equal to the number of variables in the problem)
	\param var_types types of the variables (has to be filled by function).
	*/
	virtual bool get_variables_types(Index n, VariableType* var_types);
 
	/** Pass info about linear and nonlinear variables.*/
	virtual bool get_variables_linearity(Index n, Ipopt::TNLP::LinearityType* var_types);

	/** Pass the type of the constraints (LINEAR, NON_LINEAR) to the optimizer.
	\param m size of const_types (has to be equal to the number of constraints in the problem)
	\param const_types types of the constraints (has to be filled by function).
	*/
	virtual bool get_constraints_linearity(Index m, Ipopt::TNLP::LinearityType* const_types);
	//@}  
    
	/** \name Overloaded functions defining a TNLP.
	* This group of function implement the various elements needed to define and solve a TNLP.
	* They are the same as those in a standard Ipopt NLP problem*/
	//@{
	/** Method to pass the main dimensions of the problem to Ipopt.
		\param n number of variables in problem.
		\param m number of constraints.
		\param nnz_jac_g number of nonzeroes in Jacobian of constraints system.
		\param nnz_h_lag number of nonzeroes in Hessian of the Lagrangean.
		\param index_style indicate wether arrays are numbered from 0 (C-style) or
		from 1 (Fortran).
		\return true in case of success.*/
	virtual bool get_nlp_info(Index& n, Index&m, Index& nnz_jac_g,
                            Index& nnz_h_lag, TNLP::IndexStyleEnum& index_style);
  
	/** Method to pass the bounds on variables and constraints to Ipopt. 
		\param n size of x_l and x_u (has to be equal to the number of variables in the problem)
		\param x_l lower bounds on variables (function should fill it).
		\param x_u upper bounds on the variables (function should fill it).
		\param m size of g_l and g_u (has to be equal to the number of constraints in the problem).
		\param g_l lower bounds of the constraints (function should fill it).
		\param g_u upper bounds of the constraints (function should fill it).
	\return true in case of success.*/
	virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u,
								Index m, Number* g_l, Number* g_u);
	
	/** Method to to pass the starting point for optimization to Ipopt.
	\param init_x do we initialize primals?
	\param x pass starting primal points (function should fill it if init_x is 1).
	\param m size of lambda (has to be equal to the number of constraints in the problem).
	\param init_lambda do we initialize duals of constraints? 
	\param lambda lower bounds of the constraints (function should fill it).
	\return true in case of success.*/
	virtual bool get_starting_point(Index n, bool init_x, Number* x,
									bool init_z, Number* z_L, Number* z_U,
									Index m, bool init_lambda,
									Number* lambda);
	
	/** Method which compute the value of the objective function at point x.
	\param n size of array x (has to be the number of variables in the problem).
	\param x point where to evaluate.
	\param new_x Is this the first time we evaluate functions at this point? 
	(in the present context we don't care).
	\param obj_value value of objective in x (has to be computed by the function).
	\return true in case of success.*/
	virtual bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value);
	
	/** Method which compute the gradient of the objective at a point x.
	\param n size of array x (has to be the number of variables in the problem).
	\param x point where to evaluate.
	\param new_x Is this the first time we evaluate functions at this point? 
	(in the present context we don't care).
	\param grad_f gradient of objective taken in x (function has to fill it).
	\return true in case of success.*/
	virtual bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f);
	
	/** Method which compute the value of the functions defining the constraints at a point
	x.
	\param n size of array x (has to be the number of variables in the problem).
	\param x point where to evaluate.
	\param new_x Is this the first time we evaluate functions at this point? 
	(in the present context we don't care).
	\param m size of array g (has to be equal to the number of constraints in the problem)
	\param grad_f values of the constraints (function has to fill it).
	\return true in case of success.*/
	virtual bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g);
	
	/** Method to compute the Jacobian of the functions defining the constraints.
	If the parameter values==NULL fill the arrays iCol and jRow which store the position of
	the non-zero element of the Jacobian.
	If the paramenter values!=NULL fill values with the non-zero elements of the Jacobian.
	\param n size of array x (has to be the number of variables in the problem).
	\param x point where to evaluate.
	\param new_x Is this the first time we evaluate functions at this point? 
	(in the present context we don't care).
	\param m size of array g (has to be equal to the number of constraints in the problem)
	\param grad_f values of the constraints (function has to fill it).
	\return true in case of success.*/
	virtual bool eval_jac_g(Index n, const Number* x, bool new_x,
							Index m, Index nele_jac, Index* iRow, Index *jCol,
							Number* values);
	
	/** Method to compute the Jacobian of the functions defining the constraints.
	If the parameter values==NULL fill the arrays iCol and jRow which store the position of
	the non-zero element of the Jacobian.
	If the paramenter values!=NULL fill values with the non-zero elements of the Jacobian.
	\param n size of array x (has to be the number of variables in the problem).
	\param x point where to evaluate.
	\param new_x Is this the first time we evaluate functions at this point? 
	(in the present context we don't care).
	\param m size of array g (has to be equal to the number of constraints in the problem)
	\param grad_f values of the constraints (function has to fill it).
	\return true in case of success.*/
	virtual bool eval_h(Index n, const Number* x, bool new_x,
						Number obj_factor, Index m, const Number* lambda,
						bool new_lambda, Index nele_hess, Index* iRow,
						Index* jCol, Number* values);
	
	
	/** Method called by Ipopt at the end of optimization.*/  
	virtual void finalize_solution(TMINLP::SolverReturn status,
								Index n, const Number* x, Number obj_value);
	
	//@}
	
	virtual const SosInfo * sosConstraints() const{return NULL;}
	virtual const BranchingInfo* branchingInfo() const{return NULL;}	
	
	void printSolutionAtEndOfAlgorithm() {printSol_ = true;}
  
private:
   bool printSol_;
public:
	Jipopt * ipoptProblem;
	jmethodID get_variables_types_;
	jmethodID get_variables_linearity_;
	jmethodID get_constraints_linearity_;
};

class JBonminSolver {
public:
	JBonminSolver(){}
	~JBonminSolver(){delete problem;}
public:	
	BonminSetup * bonmin;
	JBonmin * problem;	
};

JBonmin::JBonmin(JNIEnv *env, jobject solver, jint n, jint m) {

	// the solver class
	jclass solverCls = env->GetObjectClass(solver);

	// get the methods
	get_variables_types_= env->GetMethodID(solverCls,"get_variables_types","(I[I)Z");
	get_variables_linearity_= env->GetMethodID(solverCls,"get_variables_linearity","(I[I)Z");
	get_constraints_linearity_= env->GetMethodID(solverCls, "get_constraints_linearity", "(I[I)Z");
	
	if(get_variables_types_==0 || get_variables_linearity_==0 ||
		get_constraints_linearity_==0){
		std::cerr << "Expected callback methods missing on JBonmin.java" << std::endl;		
	}
}

//====================== TMINLP methods ======================

bool JBonmin::get_variables_types(Index n, Bonmin::TMINLP::VariableType* var_types) {
	jintArray bon_var_types_j = ipoptProblem->env->NewIntArray(n);
	//std::cout<<n<<"======dfd====="<<ipoptProblem->env->GetArrayLength(bon_var_types_j)<<"==="<<endl;
	if(!ipoptProblem->env->CallBooleanMethod(ipoptProblem->solver, 
		get_variables_types_, n, bon_var_types_j))
    	return false;
	jint *var_typesj = ipoptProblem->env->GetIntArrayElements(bon_var_types_j, 0);	
	for(int i = 0; i < n; i++) 
		var_types[i] = VariableType(var_typesj[i]);
	ipoptProblem->env->ReleaseIntArrayElements(bon_var_types_j, var_typesj, 0);
	return true;
}
 
bool JBonmin::get_variables_linearity(Index n, Ipopt::TNLP::LinearityType* var_types) {
	jintArray ipopt_var_types_j = ipoptProblem->env->NewIntArray(n);
	if(!ipoptProblem->env->CallBooleanMethod(this->ipoptProblem->solver, 
		get_variables_linearity_, n, ipopt_var_types_j))
    	return false;
	jint *ipopt_var_typesj = this->ipoptProblem->env->GetIntArrayElements(ipopt_var_types_j, 0);	
	for(int i = 0; i < n; i++)
		var_types[i] = Ipopt::TNLP::LinearityType(ipopt_var_typesj[i]);
	this->ipoptProblem->env->ReleaseIntArrayElements(ipopt_var_types_j, ipopt_var_typesj, 0);	
	return true;
}

bool JBonmin::get_constraints_linearity(Index m, Ipopt::TNLP::LinearityType* const_types) {
	jintArray ipopt_const_types_j = ipoptProblem->env->NewIntArray(m);
	if(!ipoptProblem->env->CallBooleanMethod(this->ipoptProblem->solver, 
		get_constraints_linearity_, m, ipopt_const_types_j))
    	return false;
	jint *ipopt_const_typesj = this->ipoptProblem->env->GetIntArrayElements(ipopt_const_types_j, 0);	
	for(int i = 0; i < m; i++)
		const_types[i] = Ipopt::TNLP::LinearityType(ipopt_const_typesj[i]);
	this->ipoptProblem->env->ReleaseIntArrayElements(ipopt_const_types_j, ipopt_const_typesj, 0);
	return true;
}

//====================== TNLP methods ===================================
bool JBonmin::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
						  Index& nnz_h_lag, TNLP::IndexStyleEnum& index_style){
	return ipoptProblem->get_nlp_info(n, m, nnz_jac_g, nnz_h_lag, index_style);	  	
}

bool JBonmin::get_bounds_info(Index n, Number *x_l, Number *x_u, Index m, Number *g_l, Number *g_u) {
	return ipoptProblem->get_bounds_info(n, x_l, x_u, m, g_l, g_u);
}

bool JBonmin::get_starting_point(Index n, bool init_x, Number* x,
                                    bool init_z, Number* z_L, Number* z_U,
                                    Index m, bool init_lambda, Number* lambda){
	return ipoptProblem->get_starting_point(n, init_x, x, init_z, z_L, z_U, m, init_lambda, lambda);
}

bool JBonmin::eval_f(Index n, const Number* x, bool new_x, Number& obj_value){
	return ipoptProblem->eval_f(n, x, new_x, obj_value);
}

bool JBonmin::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f){
	return ipoptProblem->eval_grad_f(n, x, new_x, grad_f);
}

bool JBonmin::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g){
	return ipoptProblem->eval_g(n, x, new_x, m, g);
}

bool JBonmin::eval_jac_g(Index n, const Number* x, bool new_x,
                            Index m, Index nele_jac, Index* iRow,
                            Index *jCol, Number* jac_g){
	return ipoptProblem->eval_jac_g(n, x, new_x, m, nele_jac, iRow, jCol, jac_g);
}

bool JBonmin::eval_h(Index n, const Number* x, bool new_x,
                        Number obj_factor, Index m, const Number* lambda,
                        bool new_lambda, Index nele_hess,
                        Index* iRow, Index* jCol, Number* hess) {
	return ipoptProblem->eval_h(n, x, new_x, obj_factor, m, lambda, 
								new_lambda, nele_hess, iRow, jCol, hess);
}

void JBonmin::finalize_solution(TMINLP::SolverReturn status,
								Index n, const Number* x, Number obj_value) {
	//nothing is done in this method now.
}

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL Java_org_coinor_Bonmin_CreateBonminProblem 
(JNIEnv *env, jobject obj_this, 
 jint n,  jint m,
 jint nele_jac, jint nele_hess,
 jint index_style) {
	/* create the IpoptProblem */
	Jipopt* ipoptProblem = new Jipopt(env, obj_this, n, m, nele_jac, nele_hess, index_style);
	if(ipoptProblem == NULL)
		return 0;
	jclass solverCls = env->GetObjectClass(obj_this);
	JBonmin* problem = new JBonmin(env, obj_this, n, m);	
	problem->get_variables_types_= env->GetMethodID(solverCls, "get_variables_types", "(I[I)Z");
	problem->get_variables_linearity_= env->GetMethodID(solverCls, "get_variables_linearity", "(I[I)Z");
	problem->get_constraints_linearity_= env->GetMethodID(solverCls, "get_constraints_linearity", "(I[I)Z");
	problem->ipoptProblem = ipoptProblem;
	
	BonminSetup * bonmin = new BonminSetup();
	JBonminSolver * solver = new JBonminSolver();	
	solver->problem = problem;	
	solver->bonmin = bonmin;
	solver->bonmin->initializeOptionsAndJournalist();
	//return c++ class point
	return (jlong)solver;
}

JNIEXPORT jint JNICALL Java_org_coinor_Bonmin_OptimizeTMINLP
  (JNIEnv *env, jobject obj_this, jlong pbonmin, jstring outfilename ,
  jdoubleArray xj,
jdoubleArray gj,
jdoubleArray obj_valj,
jdoubleArray mult_gj,
jdoubleArray mult_x_Lj,
jdoubleArray mult_x_Uj,

jdoubleArray callback_grad_f,
jdoubleArray callback_jac_g,
jdoubleArray callback_hess) {
	//WindowsErrorPopupBlocker();
	// cast back our class
	JBonminSolver *solver = (JBonminSolver *)pbonmin;
	JBonmin * problem = solver->problem;
	problem->ipoptProblem->env = env;
	problem->ipoptProblem->solver = obj_this;

	problem->ipoptProblem->xj = xj;
	problem->ipoptProblem->gj = gj;
	problem->ipoptProblem->fj = obj_valj;
	problem->ipoptProblem->mult_gj = mult_gj;
	problem->ipoptProblem->mult_x_Lj = mult_x_Lj;
	problem->ipoptProblem->mult_x_Uj = mult_x_Uj;

	problem->ipoptProblem->grad_fj = callback_grad_f;
	problem->ipoptProblem->jac_gj = callback_jac_g;
	problem->ipoptProblem->hessj = callback_hess;
	
	solver->bonmin->initialize(problem);
	
	//Set up done, now let's branch and bound
	try {
		Bab bb;
		bb(*(solver->bonmin));//process parameter file using Ipopt and do branch and bound using Cbc		
	} catch(TNLPSolver::UnsolvedError *E) {
		//There has been a failure to solve a problem with Ipopt.
		std::cerr<<"Ipopt has failed to solve a problem"<<std::endl;
		return -1;
	} catch(OsiTMINLPInterface::SimpleError &E) {
		std::cerr<<E.className()<<"::"<<E.methodName()
			<<std::endl
			<<E.message()<<std::endl;
		return -1;
	} catch(CoinError &E) {
		std::cerr<<E.className()<<"::"<<E.methodName()
			<<std::endl
			<<E.message()<<std::endl;
		return -1;
	}	
	return 1;
}

JNIEXPORT void JNICALL Java_org_coinor_Bonmin_FreeBonminProblem
(JNIEnv *env, jobject obj_this, jlong pbonmin) {
	// cast back our class
	JBonminSolver *solver = (JBonminSolver *)pbonmin;
	if(solver != NULL) 
		delete solver;	
}

JNIEXPORT jboolean JNICALL Java_org_coinor_Bonmin_AddBonminIntOption
(JNIEnv * env, jobject obj_this, jlong pbonmin, jstring jparname, jint jparvalue) {
	// cast back our class
	JBonminSolver *solver = (JBonminSolver *)pbonmin;
	
	const char *pparameterName = env->GetStringUTFChars(jparname, 0);
	string parameterName = pparameterName;

	// Try to apply the integer option
	jboolean ret = solver->bonmin->options()->SetIntegerValue(parameterName, jparvalue);
	
	env->ReleaseStringUTFChars(jparname, pparameterName);	
	return ret;
}

JNIEXPORT jboolean JNICALL Java_org_coinor_Bonmin_AddBonminNumOption
(JNIEnv * env, jobject obj_this, jlong pbonmin, jstring jparname, jdouble jparvalue) {  
	// cast back our class
	JBonminSolver *solver = (JBonminSolver *)pbonmin;
	
	const char *pparameterName = env->GetStringUTFChars(jparname, 0);
	string parameterName=pparameterName;

	// Try to set the real option
	jboolean ret = solver->bonmin->options()->SetNumericValue(parameterName,jparvalue);
	
	env->ReleaseStringUTFChars(jparname, pparameterName);
	
	return ret;
}

JNIEXPORT jboolean JNICALL Java_org_coinor_Bonmin_AddBonminStrOption
(JNIEnv * env, jobject obj_this, jlong pbonmin, jstring jparname, jstring jparvalue) {
	// cast back our class
	JBonminSolver *solver = (JBonminSolver *)pbonmin;
	
	const char *pparameterName = env->GetStringUTFChars(jparname, NULL);
	string parameterName = pparameterName;
	const char *pparameterValue = env->GetStringUTFChars(jparvalue, NULL);
	string parameterValue = pparameterValue;

	//parameterValue has been changed to LowerCase in Java!
	if(parameterName == "hessian_approximation" && parameterValue=="limited-memory") {
		solver->problem->ipoptProblem->using_LBFGS = true;		
	} else if(parameterName == "nlp_scaling_method" && parameterValue=="user-scaling") 
		solver->problem->ipoptProblem->using_scaling_parameters = true;			

	// Try to apply the string option
	jboolean ret = solver->bonmin->options()->SetStringValue(parameterName,parameterValue);
	env->ReleaseStringUTFChars(jparname, pparameterName);
	env->ReleaseStringUTFChars(jparname, pparameterValue);	
	return ret;
}

#ifdef __cplusplus
}
#endif