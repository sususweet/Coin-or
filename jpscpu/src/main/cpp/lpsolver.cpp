#include <jni.h>
#include "jpscpu_LinearSolver.h"

#include "CbcConfig.h"

#include "CoinPragma.hpp"

#include <cassert>
#include <iomanip>


// For Branch and bound
#include "OsiSolverInterface.hpp"
#include "CbcModel.hpp"
#include "CbcCutGenerator.hpp"
#include "CbcHeuristicLocal.hpp"
#include "OsiClpSolverInterface.hpp"

// Cuts

#include "CglGomory.hpp"
#include "CglProbing.hpp"
#include "CglKnapsackCover.hpp"
#include "CglOddHole.hpp"
#include "CglClique.hpp"
#include "CglFlowCover.hpp"
#include "CglMixedIntegerRounding.hpp"

// Heuristics

#include "CbcHeuristic.hpp"

// Methods of building

#include "CoinBuild.hpp"
#include "CoinModel.hpp"

#include "CoinTime.hpp"

#include "symphony.h"
#include <iostream>
#include <stdio.h>
#include <malloc.h>

JNIEXPORT jint JNICALL Java_jpscpu_LinearSolver_solveMlpCbc
 (JNIEnv *env, jobject obj, jint numberColumns, jint numberRows, 
 jdoubleArray jobjValue, jdoubleArray jcolumnLower,
 jdoubleArray jcolumnUpper, 
 jdoubleArray jrowLower, 
 jdoubleArray jrowUpper, jdoubleArray jelement,
 jintArray jcolumn, jintArray jstarts, jintArray jwhichInt,jdoubleArray jresult) {	
	
	jdouble * objValue = env->GetDoubleArrayElements(jobjValue, 0);	
	jdouble * columnLower = env->GetDoubleArrayElements(jcolumnLower, 0);	
	jdouble * columnUpper = env->GetDoubleArrayElements(jcolumnUpper, 0);	
	jdouble * rowLower = env->GetDoubleArrayElements(jrowLower, 0);	
	jdouble * rowUpper = env->GetDoubleArrayElements(jrowUpper, 0);	
	jdouble * element_j = env->GetDoubleArrayElements(jelement, 0);	
	jint * column_j = env->GetIntArrayElements(jcolumn, 0);	
	jint * starts = env->GetIntArrayElements(jstarts, 0);	
	jint * whichInt = env->GetIntArrayElements(jwhichInt, 0);	

	int size = env->GetArrayLength(jelement);
	double * element = new double[size];
	int * column = new int[size];
	for(int i = 0; i < size; i++) {
		element[i] = element_j[i];
		column[i] = column_j[i];
	}	
	
	/* Define your favorite OsiSolver.
	
	CbcModel clones the solver so use solver1 up to the time you pass it
	to CbcModel then use a pointer to cloned solver (model.solver())
	*/	
	OsiClpSolverInterface solver1;
	
	/* From now on we can build model in a solver independent way.
	You can add rows one at a time but for large problems this is slow so
	this example uses CoinBuild or CoinModel
	*/
	OsiSolverInterface * solver = &solver1;
	
	CoinModel build;
	// First do columns (objective and bounds)
	int i;
	for (i = 0; i < numberColumns; i++) {
		build.setColumnBounds(i,columnLower[i],columnUpper[i]);
		build.setObjective(i,objValue[i]);
	}
	// mark as integer
	size = env->GetArrayLength(jwhichInt);
	for (i = 0; i < size; i++)
		build.setInteger(whichInt[i]);
	// Now build rows
	for (i=0;i<numberRows;i++) {
		int startRow = starts[i];
		int numberInRow = starts[i+1]-starts[i];
		build.addRow(numberInRow,column+startRow,element+startRow,
					rowLower[i],rowUpper[i]);
	}  
	// add rows into solver
	solver->loadFromCoinModel(build);
	
	// Pass to solver
	CbcModel model(*solver);
	model.solver()->setHintParam(OsiDoReducePrint,true,OsiHintTry);
	
	
	// Set up some cut generators and defaults
	// Probing first as gets tight bounds on continuous
	
	CglProbing generator1;
	generator1.setUsingObjective(true);
	generator1.setMaxPass(3);
	generator1.setMaxProbe(100);
	generator1.setMaxLook(50);
	generator1.setRowCuts(3);
	//  generator1.snapshot(*model.solver());
	//generator1.createCliques(*model.solver(),2,1000,true);
	//generator1.setMode(0);
	
	CglGomory generator2;
	// try larger limit
	generator2.setLimit(300);
	
	CglKnapsackCover generator3;
	
	CglOddHole generator4;
	generator4.setMinimumViolation(0.005);
	generator4.setMinimumViolationPer(0.00002);
	// try larger limit
	generator4.setMaximumEntries(200);
	
	CglClique generator5;
	generator5.setStarCliqueReport(false);
	generator5.setRowCliqueReport(false);
	
	CglMixedIntegerRounding mixedGen;
	CglFlowCover flowGen;
	
	// Add in generators
	model.addCutGenerator(&generator1,-1,"Probing");
	model.addCutGenerator(&generator2,-1,"Gomory");
	model.addCutGenerator(&generator3,-1,"Knapsack");
	model.addCutGenerator(&generator4,-1,"OddHole");
	model.addCutGenerator(&generator5,-1,"Clique");
	model.addCutGenerator(&flowGen,-1,"FlowCover");
	model.addCutGenerator(&mixedGen,-1,"MixedIntegerRounding");
	
	OsiClpSolverInterface * osiclp = dynamic_cast< OsiClpSolverInterface*> (model.solver());
	// go faster stripes
	if (osiclp->getNumRows()<300&&osiclp->getNumCols()<500) {
		osiclp->setupForRepeatedUse(2,0);
		printf("trying slightly less reliable but faster version (? Gomory cuts okay?)\n");
		printf("may not be safe if doing cuts in tree which need accuracy (level 2 anyway)\n");
	}
	
	// Allow rounding heuristic
	
	CbcRounding heuristic1(model);
	model.addHeuristic(&heuristic1);
	
	// And local search when new solution found
	
	CbcHeuristicLocal heuristic2(model);
	model.addHeuristic(&heuristic2);
	
	// Redundant definition of default branching (as Default == User)
	//CbcBranchUserDecision branch;
	//model.setBranchingMethod(&branch);
	
	// Definition of node choice
	//CbcCompareUser compare;
	//model.setNodeComparison(compare);
	
	// Do initial solve to continuous
	model.initialSolve();
	
	// Could tune more
	model.setMinimumDrop(CoinMin(1.0,
				fabs(model.getMinimizationObjValue())*1.0e-3+1.0e-4));
	
	if (model.getNumCols()<500)
		model.setMaximumCutPassesAtRoot(-100); // always do 100 if possible
	else if (model.getNumCols()<5000)
		model.setMaximumCutPassesAtRoot(100); // use minimum drop
	else
		model.setMaximumCutPassesAtRoot(20);
	//model.setMaximumCutPasses(5);
	
	// Switch off strong branching if wanted
	// model.setNumberStrong(0);
	// Do more strong branching if small
	if (model.getNumCols() < 5000)
		model.setNumberStrong(10);
	
	model.solver()->setIntParam(OsiMaxNumIterationHotStart,100);	

	// Switch off most output
	if (model.getNumCols()<3000) {
		model.messageHandler()->setLogLevel(1);
		//model.solver()->messageHandler()->setLogLevel(0);
	} else {
		model.messageHandler()->setLogLevel(2);
		model.solver()->messageHandler()->setLogLevel(1);
	}
	
	// Do complete search	
	model.branchAndBound();
	
	int status = -1;
	if (model.getMinimizationObjValue()<1.0e50) {
		int numberColumns = model.solver()->getNumCols();
		
		const double * solution = model.solver()->getColSolution();
		jdouble * result = env->GetDoubleArrayElements(jresult, 0);	
	
		int iColumn;
		for (iColumn=0; iColumn < numberColumns; iColumn++) 
			result[iColumn] = solution[iColumn];		
		env->ReleaseDoubleArrayElements(jresult, result, 0);
		status = 1;
	}
	
	env->ReleaseDoubleArrayElements(jobjValue, objValue, 0);	
	env->ReleaseDoubleArrayElements(jcolumnLower, columnLower, 0);	
	env->ReleaseDoubleArrayElements(jcolumnUpper, columnUpper, 0);	
	env->ReleaseDoubleArrayElements(jrowLower, rowLower, 0);	
	env->ReleaseDoubleArrayElements(jrowUpper, rowUpper, 0);	
	env->ReleaseDoubleArrayElements(jelement, element_j, 0);	
	env->ReleaseIntArrayElements(jcolumn, column_j, 0);	
	env->ReleaseIntArrayElements(jstarts, starts, 0);	
	env->ReleaseIntArrayElements(jwhichInt, whichInt, 0);	
	
	delete[] element;
	delete[] column;
	return status;
}

JNIEXPORT jint JNICALL Java_jpscpu_LinearSolver_solveMlpSym
 (JNIEnv *env, jobject obj, jint numberColumns, jint numberRows, 
 jdoubleArray jobjValue, jdoubleArray jcolumnLower,
 jdoubleArray jcolumnUpper, 
 jdoubleArray jrowLower, 
 jdoubleArray jrowUpper, jdoubleArray jelement,
 jintArray jcolumn, jintArray jstarts, jintArray jwhichInt,jdoubleArray jresult) {	
	
	jdouble * objValue = env->GetDoubleArrayElements(jobjValue, 0);	
	jdouble * columnLower = env->GetDoubleArrayElements(jcolumnLower, 0);	
	jdouble * columnUpper = env->GetDoubleArrayElements(jcolumnUpper, 0);	
	jdouble * rowLower = env->GetDoubleArrayElements(jrowLower, 0);	
	jdouble * rowUpper = env->GetDoubleArrayElements(jrowUpper, 0);	
	jdouble * element_j = env->GetDoubleArrayElements(jelement, 0);	
	jint * column_j = env->GetIntArrayElements(jcolumn, 0);	
	jint * starts = env->GetIntArrayElements(jstarts, 0);	
	jint * whichInt = env->GetIntArrayElements(jwhichInt, 0);	

	int size = env->GetArrayLength(jelement);
	double * element = (double *) malloc(sizeof(double) * size);
	int * column = (int *) malloc(sizeof(int) * size);
	for(int i = 0; i < size; i++) {
		element[i] = element_j[i];
		column[i] = column_j[i];
	}	
	
	sym_environment *sym_env = sym_open_environment();

	int n_cols = numberColumns; //number of columns
	double * objective    = 
		(double *) malloc(sizeof(double) * n_cols);//the objective coefficients
	double * col_lb       = 
		(double *) malloc(sizeof(double) * n_cols);//the column lower bounds
	double * col_ub       = 
		(double *) malloc(sizeof(double) * n_cols);//the column upper bounds
		
	for(int i = 0; i < n_cols; i++) {
		objective[i] = objValue[i];
		col_lb[i] = columnLower[i];
		col_ub[i] = columnUpper[i];
	}
	
	int n_rows = numberRows;
	char * row_sense = 
		(char *) malloc (sizeof(char) * n_rows); //the row senses
	double * row_rhs = 
		(double *) malloc (sizeof(double) * n_rows); //the row right-hand-sides
	double * row_range =
		(double *) malloc (sizeof(double) * n_rows); //the row ranges  
	for(int i = 0; i < n_rows; i++) {
		row_range[i] = 0.0;
		if(rowLower[i] == rowUpper[i]) {
			row_sense[i] = 'E';
			row_rhs[i] = rowUpper[i];
		} else if(rowLower[i] <= -1e15) {
			row_sense[i] = 'L';
			row_rhs[i] = rowUpper[i];
		} else if(rowUpper[i] >= 1e15) {
			row_sense[i] = 'G';
			row_rhs[i] = rowLower[i];
		} else {
			row_sense[i] = 'R';
			row_rhs[i] = rowUpper[i];
			row_range[i] = rowUpper[i] - rowLower[i];
		}
	}
	
	/* Constraint matrix definitions */
	int * start = (int *) malloc (sizeof(int) * (n_cols + 1)); 
	for(int i = 0; i < n_cols + 1; i++)
		start[i] = starts[i];
	
	//define the integer variables	
	char * int_vars = (char *) malloc (sizeof(char) * n_cols);
	
	size = env->GetArrayLength(jwhichInt);
	for(int i = 0; i < n_cols; i++)
		int_vars[i] = FALSE;
	for(int i = 0; i < size; i++)
		int_vars[whichInt[i]] = TRUE;
		
	//load the problem to environment
	sym_explicit_load_problem(sym_env, n_cols, n_rows, start, column, element, col_lb, 
					col_ub, int_vars, objective, NULL, row_sense, 
					row_rhs, row_range, TRUE);
	//solve the integer program
	int status = sym_solve(sym_env);
	
	if (status == TM_OPTIMAL_SOLUTION_FOUND) {	
		//get, print the solution
		double * solution = (double *) malloc (sizeof(double) * n_cols);
		sym_get_col_solution(sym_env, solution);
	
		jdouble * result = env->GetDoubleArrayElements(jresult, 0);	
	
		int iColumn;
		for (iColumn = 0; iColumn < numberColumns; iColumn++) 
			result[iColumn] = solution[iColumn];		
		env->ReleaseDoubleArrayElements(jresult, result, 0);
		
		free(solution);
		status = 1;
	} else {
		status = -1;
		std::cout<<"Failed to solve MLP by SYMPHONY."<<std::endl;
	}
	
	//free the memory
	sym_close_environment(sym_env);
	if(objective){free(objective);}
	if(col_lb)   {free(col_lb);}
	if(col_ub)   {free(col_ub);}
	if(row_rhs)  {free(row_rhs);}
	if(row_sense){free(row_sense);}
	if(row_range){free(row_range);}
	if(column)    {free(column);}
	if(start)    {free(start);}
	if(element)    {free(element);}
	if(int_vars) {free(int_vars);}	

	env->ReleaseDoubleArrayElements(jobjValue, objValue, 0);
	env->ReleaseDoubleArrayElements(jcolumnLower, columnLower, 0);	
	env->ReleaseDoubleArrayElements(jcolumnUpper, columnUpper, 0);	
	env->ReleaseDoubleArrayElements(jrowLower, rowLower, 0);	
	env->ReleaseDoubleArrayElements(jrowUpper, rowUpper, 0);	
	env->ReleaseDoubleArrayElements(jelement, element_j, 0);	
	env->ReleaseIntArrayElements(jcolumn, column_j, 0);	
	env->ReleaseIntArrayElements(jstarts, starts, 0);	
	env->ReleaseIntArrayElements(jwhichInt, whichInt, 0);

	return status;
}