#include <jni.h>
#include "jpscpu_LinearSolver.h"

#include "slu_ddefs.h"
#include "sstream"

using namespace std;

// use simple driver DGSSV to solve a linear system one time.
double* solve0(int m, int n, int nnz, double* a, int* asub, int* xa, double* b) {
	SuperMatrix A;
    int      *perm_c; /* column permutation vector */
    int      *perm_r; /* row permutations from partial pivoting */
    SuperMatrix L;      /* factor L */   
    SuperMatrix U;      /* factor U */
    SuperMatrix B;
    int      nrhs, info;
    double   *rhs, *result;
    superlu_options_t options;
    SuperLUStat_t stat;
   
    dCreate_CompCol_Matrix(&A, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);
    
	nrhs = 1;
    if ( !(rhs = doubleMalloc(m * nrhs)) ) ABORT("Malloc fails for rhs[].");
    if ( !(result = doubleMalloc(m * nrhs)) ) ABORT("Malloc fails for result[].");
    dCreate_Dense_Matrix(&B, m, nrhs, rhs, m, SLU_DN, SLU_D, SLU_GE);
    
	for (int i = 0; i < m; ++i) 
		rhs[i] = b[i];

    if ( !(perm_c = intMalloc(n)) ) ABORT("Malloc fails for perm_c[].");
    if ( !(perm_r = intMalloc(m)) ) ABORT("Malloc fails for perm_r[].");

	/* Set the default input options. */
	set_default_options(&options);
	options.ColPerm = NATURAL;
		
    /* Initialize the statistics variables. */
    StatInit(&stat);
    
    dgssv(&options, &A, perm_c, perm_r, &L, &U, &B, &stat, &info);    
    if ( info == 0 ) {
		for (int i = 0; i < n; i++)
			result[i] = ((double*)((DNformat*)B.Store)->nzval)[i]; 
    } else {
		printf("dgssv() error returns INFO= %d\n", info);		
    }
    //if ( options.PrintStat )
	//	StatPrint(&stat);
    StatFree(&stat);

	//dPrint_CompCol_Matrix("A", &A);
	//dPrint_CompCol_Matrix("U", &U);
	//dPrint_SuperNode_Matrix("L", &L);
	
    SUPERLU_FREE (rhs);
    SUPERLU_FREE (perm_r);
    SUPERLU_FREE (perm_c);
    Destroy_CompCol_Matrix(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperNode_Matrix(&L);
    Destroy_CompCol_Matrix(&U);
	
	return result;
}

//use DGSSVX to solve a linear system.
double* solve1(int m, int n, int nnz, double* a, int* asub, int* xa, double* b) {

	char           equed[1];
	yes_no_t       equil;
	trans_t        trans;
	SuperMatrix    A, L, U;
	SuperMatrix    B, X;
	GlobalLU_t	   Glu; /* facilitate multiple factorizations with
                               SamePattern_SameRowPerm                */
	int            *perm_r; /* row permutations from partial pivoting */
	int            *perm_c; /* column permutation vector */
	int            *etree;
	void           *work;
	int            info, lwork, nrhs;
	int            i;
	double         *rhsb, *rhsx, *result;
	double         *R, *C;
	double         *ferr, *berr;
	double         u, rpg, rcond;
	mem_usage_t    mem_usage;
	superlu_options_t options;
	SuperLUStat_t stat;
	// Defaults
	lwork = 0;
	nrhs  = 1;
	equil = YES;
	u     = 1.0;
	trans = NOTRANS;
	set_default_options(&options);
	
	options.Equil = equil;
	options.DiagPivotThresh = u;
	options.Trans = trans;
	
	if ( lwork > 0 ) {
		work = SUPERLU_MALLOC(lwork);
		if ( !work ) {
			ABORT("DLINSOLX: cannot allocate work[]");
		}
	}
	
	/* Initialize matrix A. */
	dCreate_CompCol_Matrix(&A, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);
	
	/* Create right-hand side matrix B and X. */
	if ( !(rhsb = doubleMalloc(m * nrhs)) ) ABORT("Malloc fails for rhsb[].");
	if ( !(rhsx = doubleMalloc(m * nrhs)) ) ABORT("Malloc fails for rhsx[].");
	if ( !(result = doubleMalloc(n * nrhs)) ) ABORT("Malloc fails for result[].");
	for (i = 0; i < m; ++i) {
		rhsb[i] = b[i];
		rhsx[i] = b[i];
	}
	dCreate_Dense_Matrix(&B, m, nrhs, rhsb, m, SLU_DN, SLU_D, SLU_GE);
	dCreate_Dense_Matrix(&X, m, nrhs, rhsx, m, SLU_DN, SLU_D, SLU_GE);
	
	if ( !(perm_c = intMalloc(n)) ) ABORT("Malloc fails for perm_c[].");
	if ( !(perm_r = intMalloc(m)) ) ABORT("Malloc fails for perm_r[].");
	if ( !(etree = intMalloc(n)) ) ABORT("Malloc fails for etree[].");
	if ( !(R = (double *) SUPERLU_MALLOC(A.nrow * sizeof(double))) )
		ABORT("SUPERLU_MALLOC fails for R[].");
	if ( !(C = (double *) SUPERLU_MALLOC(A.ncol * sizeof(double))) )
		ABORT("SUPERLU_MALLOC fails for C[].");
	if ( !(ferr = (double *) SUPERLU_MALLOC(nrhs * sizeof(double))) )
		ABORT("SUPERLU_MALLOC fails for ferr[].");
	if ( !(berr = (double *) SUPERLU_MALLOC(nrhs * sizeof(double))) )
		ABORT("SUPERLU_MALLOC fails for berr[].");
	StatInit(&stat);
	
	/* ------------------------------------------------------------
		WE SOLVE THE LINEAR SYSTEM FOR THE FIRST TIME: AX = B
		------------------------------------------------------------*/
	dgssvx(&options, &A, perm_c, perm_r, etree, equed, R, C,
	&L, &U, work, lwork, &B, &X, &rpg, &rcond, ferr, berr,
	&Glu, &mem_usage, &stat, &info);
	
	if ( info == 0 || info == n+1 ) {
        for (int i = 0; i < n; i++)
			result[i] = ((double*)((DNformat*)X.Store)->nzval)[i]; 		
    } else if ( info > 0 && lwork == -1 ) {
        printf("** Estimated memory: %d bytes\n", info - n);
    }

    //if ( options.PrintStat )
	//	StatPrint(&stat);
    StatFree(&stat);

    SUPERLU_FREE (rhsb);
    SUPERLU_FREE (rhsx);
    SUPERLU_FREE (etree);
    SUPERLU_FREE (perm_r);
    SUPERLU_FREE (perm_c);
    SUPERLU_FREE (R);
    SUPERLU_FREE (C);
    SUPERLU_FREE (ferr);
    SUPERLU_FREE (berr);
    Destroy_CompCol_Matrix(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperMatrix_Store(&X);
    if ( lwork == 0 ) {
        Destroy_SuperNode_Matrix(&L);
        Destroy_CompCol_Matrix(&U);
    } else if ( lwork > 0 ) {
        SUPERLU_FREE(work);
    }
	
	return result;
}

//use DGSSVX to solve Ax = b first time.
// 将A分解结果可重用的部分存入 perm_c 和 etree
double* solve2(int m,int n,int nnz,double* a,int* asub,int* xa,double* b,int* perm_c,int *etree){

	char           equed[1];
	yes_no_t       equil;
	trans_t        trans;
	SuperMatrix    A, L, U;
	SuperMatrix    B, X;
	GlobalLU_t	   Glu; /* facilitate multiple factorizations with
                                   SamePattern_SameRowPerm            */
	int            *perm_r; /* row permutations from partial pivoting */
	void           *work;
	int            info, lwork, nrhs;
	int            i;
	double         *rhsb, *rhsx, *sol;
	double         *R, *C;
	double         *ferr, *berr;
	double         u, rpg, rcond;

	mem_usage_t    mem_usage;
	superlu_options_t options;
	SuperLUStat_t stat;

	// Defaults
	lwork = 0;
	nrhs  = 1;
	equil = YES;
	u     = 1.0;
	trans = NOTRANS;
	set_default_options(&options);

	options.Equil = equil;
	options.DiagPivotThresh = u;
	options.Trans = trans;

    /* Add more functionalities that the defaults. */
    options.PivotGrowth = YES;    /* Compute reciprocal pivot growth */
    options.ConditionNumber = YES;/* Compute reciprocal condition number */
    options.IterRefine = SLU_DOUBLE;  /* Perform double-precision refinement */

	if ( lwork > 0 ) {     
		work = SUPERLU_MALLOC(lwork);
		if ( !work ) {
			ABORT("DLINSOLX: cannot allocate work[]");
		}
	}

	/* Initialize matrix A. */
	dCreate_CompCol_Matrix(&A, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);
  
	/* Create right-hand side matrix B and X. */
	if ( !(rhsb = doubleMalloc(m * nrhs)) ) ABORT("Malloc fails for rhsb[].");
    if ( !(rhsx = doubleMalloc(m * nrhs)) ) ABORT("Malloc fails for rhsx[].");
    if ( !(sol = doubleMalloc(m * nrhs)) ) ABORT("Malloc fails for sol[].");
	
	for (i = 0; i < m; ++i) {
		rhsb[i] = b[i];
		rhsx[i] = b[i];
	}
    dCreate_Dense_Matrix(&B, m, nrhs, rhsb, m, SLU_DN, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&X, m, nrhs, rhsx, m, SLU_DN, SLU_D, SLU_GE);

    if ( !(perm_r = intMalloc(m)) ) ABORT("Malloc fails for perm_r[].");
    if ( !(R = (double *) SUPERLU_MALLOC(A.nrow * sizeof(double))) )
        ABORT("SUPERLU_MALLOC fails for R[].");
    if ( !(C = (double *) SUPERLU_MALLOC(A.ncol * sizeof(double))) )
        ABORT("SUPERLU_MALLOC fails for C[].");
    if ( !(ferr = (double *) SUPERLU_MALLOC(nrhs * sizeof(double))) )
        ABORT("SUPERLU_MALLOC fails for ferr[].");
    if ( !(berr = (double *) SUPERLU_MALLOC(nrhs * sizeof(double))) )
        ABORT("SUPERLU_MALLOC fails for berr[].");

	/* Initialize the statistics variables. */
    StatInit(&stat);

    /* ------------------------------------------------------------
       WE SOLVE THE LINEAR SYSTEM FOR THE FIRST TIME: AX = B
       ------------------------------------------------------------*/
    dgssvx(&options, &A, perm_c, perm_r, etree, equed, R, C,
           &L, &U, work, lwork, &B, &X, &rpg, &rcond, ferr, berr,
           &Glu, &mem_usage, &stat, &info);

    if ( info == 0 || info == n+1 ) {
        for (int i = 0; i < n; i++)
			sol[i] = ((double*)((DNformat*)X.Store)->nzval)[i];   
    } else if ( info > 0 && lwork == -1 ) {
        printf("** Estimated memory: %d bytes\n", info - n);
    }

	//if ( options.PrintStat )
	//	StatPrint(&stat);
    StatFree(&stat);

	SUPERLU_FREE (rhsb);
    SUPERLU_FREE (rhsx);
    SUPERLU_FREE (perm_r);
    SUPERLU_FREE (R);
    SUPERLU_FREE (C);
    SUPERLU_FREE (ferr);
    SUPERLU_FREE (berr);
    Destroy_CompCol_Matrix(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperMatrix_Store(&X);
    if ( lwork == 0 ) {
        Destroy_SuperNode_Matrix(&L);
        Destroy_CompCol_Matrix(&U);
    } else if ( lwork > 0 ) {
        SUPERLU_FREE(work);
    }
	return sol;
}

//use DGSSVX to solve systems repeatedly with the same sparsity pattern and similar numerical values as matrix A.
// 也就是所重用了 perm_c and etree
double* solve3(int m, int n, int nnz, double* a, int* asub, int* xa,double* b,int* perm_c,int *etree) {

	char           equed[1];
	yes_no_t       equil;
	trans_t        trans;
	SuperMatrix    A1, L, U;
	SuperMatrix    B, X;
    GlobalLU_t	   Glu; /* facilitate multiple factorizations with
                                   SamePattern_SameRowPerm            */
	int            *perm_r; /* row permutations from partial pivoting */
	void           *work;
	int            info, lwork, nrhs;
	int            i;
	double         *rhsb, *rhsx, *sol;
	double         *R, *C;
	double         *ferr, *berr;
	double         u, rpg, rcond;

	mem_usage_t    mem_usage;
	superlu_options_t options;
	SuperLUStat_t stat;

	// Defaults
	lwork = 0;
	nrhs  = 1;
	equil = YES;
	u     = 1.0;
	trans = NOTRANS;
	set_default_options(&options);

	options.Equil = equil;
	options.DiagPivotThresh = u;
	options.Trans = trans;

    /* Add more functionalities that the defaults. */
    options.PivotGrowth = YES;    /* Compute reciprocal pivot growth */
    options.ConditionNumber = YES;/* Compute reciprocal condition number */
    options.IterRefine = SLU_DOUBLE;  /* Perform double-precision refinement */

	if ( lwork > 0 ) {       
		work = SUPERLU_MALLOC(lwork);
		if ( !work ) {
			ABORT("DLINSOLX: cannot allocate work[]");
		}
	}


	/* Initialize matrix A1. */
	dCreate_CompCol_Matrix(&A1, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);
  
	/* Create right-hand side matrix B and X. */
	if ( !(rhsb = doubleMalloc(m * nrhs)) ) ABORT("Malloc fails for rhsb[].");
    if ( !(rhsx = doubleMalloc(m * nrhs)) ) ABORT("Malloc fails for rhsx[].");
	if ( !(sol = doubleMalloc(n * nrhs)) ) ABORT("Malloc fails for rhsx[].");

	for (i = 0; i < m; ++i) {
		rhsb[i] = b[i];
		rhsx[i] = b[i];
	}
    dCreate_Dense_Matrix(&B, m, nrhs, rhsb, m, SLU_DN, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&X, m, nrhs, rhsx, m, SLU_DN, SLU_D, SLU_GE);

    if ( !(perm_r = intMalloc(m)) ) ABORT("Malloc fails for perm_r[].");
    if ( !(R = (double *) SUPERLU_MALLOC(A1.nrow * sizeof(double))) )
        ABORT("SUPERLU_MALLOC fails for R[].");
    if ( !(C = (double *) SUPERLU_MALLOC(A1.ncol * sizeof(double))) )
        ABORT("SUPERLU_MALLOC fails for C[].");
    if ( !(ferr = (double *) SUPERLU_MALLOC(nrhs * sizeof(double))) )
        ABORT("SUPERLU_MALLOC fails for ferr[].");
    if ( !(berr = (double *) SUPERLU_MALLOC(nrhs * sizeof(double))) )
        ABORT("SUPERLU_MALLOC fails for berr[].");


	options.Fact = SamePattern;		//稀疏结构相同，复用perm_c

	/* Initialize the statistics variables. */
    StatInit(&stat);

    /* ------------------------------------------------------------
       WE SOLVE THE LINEAR SYSTEM FOR THE FIRST TIME: A1X = B
       ------------------------------------------------------------*/
    dgssvx(&options, &A1, perm_c, perm_r, etree, equed, R, C,
           &L, &U, work, lwork, &B, &X, &rpg, &rcond, ferr, berr,
           &Glu, &mem_usage, &stat, &info);
	if ( info == 0 || info == n+1 ) {
		for (int i = 0; i < n; i++)
			sol[i] = ((double*)((DNformat*)X.Store)->nzval)[i];        
    } else if ( info > 0 && lwork == -1 ) {
        printf("** Estimated memory: %d bytes\n", info - n);
    }

	//if ( options.PrintStat )
	//	StatPrint(&stat);
    StatFree(&stat);

	SUPERLU_FREE (rhsb);
    SUPERLU_FREE (rhsx);
    SUPERLU_FREE (perm_r);
    SUPERLU_FREE (R);
    SUPERLU_FREE (C);
    SUPERLU_FREE (ferr);
    SUPERLU_FREE (berr);
    Destroy_CompCol_Matrix(&A1);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperMatrix_Store(&X);
    if ( lwork == 0 ) {
        Destroy_SuperNode_Matrix(&L);
        Destroy_CompCol_Matrix(&U);
    } else if ( lwork > 0 ) {
        SUPERLU_FREE(work);
    }
	return sol;
}

// use DGSSVX to factorize A first, then solve the system later.
// 保存了A分解之后的信息，并将将分解后的A矩阵非零元赋值到ja中
double* solve4(int m, int n, int nnz, double* a, int* asub, int* xa,double* b, int* perm_c, int* perm_r, int *etree, double* R, double* C, SuperMatrix *L, SuperMatrix *U, jdouble *ja, jint* jequed) {
    char           equed[1];
    yes_no_t       equil;
    trans_t        trans;
    SuperMatrix    A;
    SuperMatrix    B, X;
    GlobalLU_t	   Glu; /* facilitate multiple factorizations with
                           SamePattern_SameRowPerm            */
	double         *rhsb, *rhsx, *sol;
    void           *work;
    int            info, lwork, nrhs;
    int            i;
    double         *ferr, *berr;
    double         u, rpg, rcond;
    mem_usage_t    mem_usage;
    superlu_options_t options;
    SuperLUStat_t stat;
    extern void    parse_command_line();

    /* Defaults */
    lwork = 0;
    nrhs  = 1;
    equil = YES;	
    u     = 1.0;
    trans = NOTRANS;

    /* Set the default values for options argument:
	options.Fact = DOFACT;
        options.Equil = YES;
    	options.ColPerm = COLAMD;
	options.DiagPivotThresh = 1.0;
    	options.Trans = NOTRANS;
    	options.IterRefine = NOREFINE;
    	options.SymmetricMode = NO;
    	options.PivotGrowth = NO;
    	options.ConditionNumber = NO;
    	options.PrintStat = YES;
    */
    set_default_options(&options);

    options.Equil = equil;
    options.DiagPivotThresh = u;
    options.Trans = trans;
    
    if ( lwork > 0 ) {
		work = SUPERLU_MALLOC(lwork);
		if ( !work ) {
			ABORT("DLINSOLX: cannot allocate work[]");
		}
    }
    
    dCreate_CompCol_Matrix(&A, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);
    
    if ( !(rhsb = doubleMalloc(m * nrhs)) ) ABORT("Malloc fails for rhsb[].");
    if ( !(rhsx = doubleMalloc(m * nrhs)) ) ABORT("Malloc fails for rhsx[].");
    if ( !(sol = doubleMalloc(n * nrhs)) ) ABORT("Malloc fails for rhsx[].");

	for (i = 0; i < m; ++i) {
		rhsb[i] = b[i];
		rhsx[i] = b[i];
	}
	dCreate_Dense_Matrix(&B, m, nrhs, rhsb, m, SLU_DN, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&X, m, nrhs, rhsx, m, SLU_DN, SLU_D, SLU_GE);
      
    if ( !(ferr = (double *) SUPERLU_MALLOC(nrhs * sizeof(double))) )
        ABORT("SUPERLU_MALLOC fails for ferr[].");
    if ( !(berr = (double *) SUPERLU_MALLOC(nrhs * sizeof(double))) ) 
        ABORT("SUPERLU_MALLOC fails for berr[].");

    /* Initialize the statistics variables. */
    StatInit(&stat);
    
    /* ONLY PERFORM THE LU DECOMPOSITION */
    B.ncol = 0;  /* Indicate not to solve the system */
    dgssvx(&options, &A, perm_c, perm_r, etree, equed, R, C,
           L, U, work, lwork, &B, &X, &rpg, &rcond, ferr, berr,
           &Glu, &mem_usage, &stat, &info);
	StatFree(&stat);		
	
	/* ------------------------------------------------------------
       NOW WE SOLVE THE LINEAR SYSTEM USING THE FACTORED FORM OF A.
       ------------------------------------------------------------*/
    options.Fact = FACTORED; // Indicate the factored form of A is supplied. 
    B.ncol = nrhs;  // Set the number of right-hand side 
	
    // Initialize the statistics variables.
    StatInit(&stat);
	
	/*
	dPrint_CompCol_Matrix("A", &A);
	dPrint_Dense_Matrix("B", &B);
	dPrint_CompCol_Matrix("U", U);
	dPrint_SuperNode_Matrix("L", L);
	print_int_vec("\nperm_c", n, perm_c);	
	print_int_vec("\nperm_r", m, perm_r);	
	print_int_vec("\netree", n, etree);	
	print_double_vec("\nR", m, R);	
	print_double_vec("\nC", n, C);
	printf("%c \n", equed[0]);	
    */	
    dgssvx(&options, &A, perm_c, perm_r, etree, equed, R, C,
           L, U, work, lwork, &B, &X, &rpg, &rcond, ferr, berr,
           &Glu, &mem_usage, &stat, &info);
	if ( info == 0 || info == n + 1 ) {
       for (int i = 0; i < n; i++)
			sol[i] = ((double*)((DNformat*)X.Store)->nzval)[i];   
	} else if ( info > 0 && lwork == -1 ) {
        printf("** Estimated memory: %d bytes\n", info - n);
    }
	
	//if ( options.PrintStat )
	//	StatPrint(&stat);
	
	StatFree(&stat);
	
	//将分解后的A矩阵非零元赋值到ja中
	for(int i = 0; i < nnz; i++)
		ja[i] = a[i];	
	jequed[0] = equed[0];
	
    SUPERLU_FREE (rhsb);
    SUPERLU_FREE (rhsx);
    SUPERLU_FREE (ferr);
    SUPERLU_FREE (berr);
    Destroy_CompCol_Matrix(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperMatrix_Store(&X);
    if ( lwork > 0 ) 
        SUPERLU_FREE(work);   
	return sol;
}

//参数提供了分解后A矩阵的信息
//配合solve4一起使用,适合求解A不变,b变化之后多次求解的场合
double* solve5(int m, int n, int nnz, double* a, int* asub, int* xa,double* b, int* perm_c, int* perm_r, int *etree, double* R, double* C, SuperMatrix *L, SuperMatrix *U, jint* jequed) {
    char           equed[1];
    yes_no_t       equil;
    trans_t        trans;
    SuperMatrix    A;
    SuperMatrix    B, X;
    GlobalLU_t	   Glu; // facilitate multiple factorizations with SamePattern_SameRowPerm
	double         *rhsb, *rhsx, *sol;
    void           *work;
    int            info, lwork, nrhs;
    int            i;
    double         *ferr, *berr;
    double         u, rpg, rcond;
    mem_usage_t    mem_usage;
    superlu_options_t options;
    SuperLUStat_t stat;
  
    /* Defaults */
	equed[0] = jequed[0]; //这个参数十分重要，但是还没有搞清楚它的具体作用
    lwork = 0;
    nrhs  = 1;
    equil = YES;	
    u     = 1.0;
    trans = NOTRANS;

   
    set_default_options(&options);

    options.Equil = equil;
    options.DiagPivotThresh = u;
    options.Trans = trans;
    
    if ( lwork > 0 ) {
		work = SUPERLU_MALLOC(lwork);
		if ( !work ) {
			ABORT("DLINSOLX: cannot allocate work[]");
		}
    }
    
    dCreate_CompCol_Matrix(&A, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);
    
    if ( !(rhsb = doubleMalloc(m * nrhs)) ) ABORT("Malloc fails for rhsb[].");
    if ( !(rhsx = doubleMalloc(m * nrhs)) ) ABORT("Malloc fails for rhsx[].");
    if ( !(sol = doubleMalloc(n * nrhs)) ) ABORT("Malloc fails for rhsx[].");

	for (i = 0; i < m; ++i) {
		rhsb[i] = b[i];
		rhsx[i] = b[i];
	}
	dCreate_Dense_Matrix(&B, m, nrhs, rhsb, m, SLU_DN, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&X, m, nrhs, rhsx, m, SLU_DN, SLU_D, SLU_GE);
      
    if ( !(ferr = (double *) SUPERLU_MALLOC(nrhs * sizeof(double))) )
        ABORT("SUPERLU_MALLOC fails for ferr[].");
    if ( !(berr = (double *) SUPERLU_MALLOC(nrhs * sizeof(double))) ) 
        ABORT("SUPERLU_MALLOC fails for berr[].");

    /* ------------------------------------------------------------
       NOW WE SOLVE THE LINEAR SYSTEM USING THE FACTORED FORM OF A.
       ------------------------------------------------------------*/
    options.Fact = FACTORED; /* Indicate the factored form of A is supplied. */
    B.ncol = nrhs;  /* Set the number of right-hand side */

    /* Initialize the statistics variables. */
    StatInit(&stat);

	/*
	dPrint_CompCol_Matrix("A", &A);
	dPrint_Dense_Matrix("B", &B);
	dPrint_CompCol_Matrix("U", U);
	dPrint_SuperNode_Matrix("L", L);
	print_int_vec("\nperm_c", n, perm_c);	
	print_int_vec("\nperm_r", m, perm_r);	
	print_int_vec("\netree", n, etree);	
	print_double_vec("\nR", m, R);	
	print_double_vec("\nC", n, C);
	printf("%c \n", equed[0]);	
    */
	dgssvx(&options, &A, perm_c, perm_r, etree, equed, R, C,
           L, U, work, lwork, &B, &X, &rpg, &rcond, ferr, berr,
           &Glu, &mem_usage, &stat, &info);
   
    if ( info == 0 || info == n + 1 ) {
       for (int i = 0; i < n; i++)
			sol[i] = ((double*)((DNformat*)X.Store)->nzval)[i];   
	} else if ( info > 0 && lwork == -1 ) {
        printf("** Estimated memory: %d bytes\n", info - n);
    }
	
	StatFree(&stat);
	
    SUPERLU_FREE (rhsb);
    SUPERLU_FREE (rhsx);
    SUPERLU_FREE (ferr);
    SUPERLU_FREE (berr);
    Destroy_CompCol_Matrix(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperMatrix_Store(&X);
    if ( lwork > 0 ) 
        SUPERLU_FREE(work);   
	return sol;
}

//======================== jni方法开始 ===========================
JNIEXPORT jdoubleArray JNICALL Java_jpscpu_LinearSolver_solve0
  (JNIEnv * env, jobject obj, jint m, jint n, jint nnz, jdoubleArray a, jintArray asub, jintArray xa, jdoubleArray b) {
	jint *jasub, *jxa;
	jdouble *ja, *jb;
	double sb[n];
	double *sa;
	int *sasub, *sxa;	
	
	if (!(sa = doubleMalloc(nnz))) ABORT("Malloc fails for a[].");
	if (!(sasub = intMalloc(nnz))) ABORT("Malloc fails for asub[].");
	if (!(sxa = intMalloc(n+1))) ABORT("Malloc fails for xa[].");
	
	jasub = env->GetIntArrayElements(asub, 0);
	jxa = env->GetIntArrayElements(xa, 0);
	ja = env->GetDoubleArrayElements(a, 0);
	jb = env->GetDoubleArrayElements(b, 0);
	
	for(int i = 0; i< nnz; i++) {
		sa[i] = (double)ja[i];
		sasub[i] = (int)jasub[i];
	}
	for(int j = 0; j < n; j++) {
		sb[j] = (double)jb[j];
	}
	for(int k = 0; k < n + 1; k++) {
		sxa[k] = (int)jxa[k];
	}
	double* res = solve0(m, n, nnz, sa, sasub, sxa, sb);	
	
	env->ReleaseDoubleArrayElements(a, ja, JNI_ABORT);
	env->ReleaseIntArrayElements(asub, jasub, JNI_ABORT);	
	env->ReleaseIntArrayElements(xa, jxa, JNI_ABORT);
	env->ReleaseDoubleArrayElements(b, jb, JNI_ABORT);	
	
	env->SetDoubleArrayRegion(b, 0, n, (const jdouble*)res);
	
	SUPERLU_FREE (res);
	return b;
}

JNIEXPORT jdoubleArray JNICALL Java_jpscpu_LinearSolver_solve1
  (JNIEnv * env, jobject obj, jint m, jint n, jint nnz, jdoubleArray a, jintArray asub, jintArray xa, jdoubleArray b) {
	jint *jasub, *jxa;
	jdouble *ja, *jb;
	double sb[n];
	double *sa;
	int *sasub, *sxa;	
	
	if (!(sa = doubleMalloc(nnz))) ABORT("Malloc fails for a[].");
	if (!(sasub = intMalloc(nnz))) ABORT("Malloc fails for asub[].");
	if (!(sxa = intMalloc(n+1))) ABORT("Malloc fails for xa[].");
	
	jasub = env->GetIntArrayElements(asub, 0);
	jxa = env->GetIntArrayElements(xa, 0);
	ja = env->GetDoubleArrayElements(a, 0);
	jb = env->GetDoubleArrayElements(b, 0);
	
	for(int i = 0; i< nnz; i++){
		sa[i] = (double)ja[i];
		sasub[i] = (int)jasub[i];
	}
	for(int j = 0; j < n; j++) {
		sb[j] = (double)jb[j];
	}
	for(int k = 0; k < n + 1; k++) {
		sxa[k] = (int)jxa[k];
	}
	double* res = solve1(m, n, nnz, sa, sasub, sxa, sb);	
	
	env->ReleaseDoubleArrayElements(a, ja, JNI_ABORT);
	env->ReleaseIntArrayElements(asub, jasub, JNI_ABORT);	
	env->ReleaseIntArrayElements(xa, jxa, JNI_ABORT);	
	env->ReleaseDoubleArrayElements(b, jb, JNI_ABORT);
	
	env->SetDoubleArrayRegion(b, 0, n, (const jdouble*)res);
	
	SUPERLU_FREE (res);
	return b;
}

JNIEXPORT jdoubleArray JNICALL Java_jpscpu_LinearSolver_solve2
  (JNIEnv *env, jobject obj, jint m, jint n, jint nnz, jdoubleArray a, jintArray asub, jintArray xa, jdoubleArray b, jintArray perm_c, jintArray etree){
    jint *jasub, *jxa;
    jdouble *ja, *jb;

    double sb[n];
    int sperm_c[n];
    int setree[n];
	double *sa;
	int *sasub, *sxa;	
	
	if (!(sa = doubleMalloc(nnz))) ABORT("Malloc fails for a[].");
	if (!(sasub = intMalloc(nnz))) ABORT("Malloc fails for asub[].");
	if (!(sxa = intMalloc(n+1))) ABORT("Malloc fails for xa[].");
	
    jasub = env->GetIntArrayElements(asub, 0);
    jxa = env->GetIntArrayElements(xa, 0);
    ja = env->GetDoubleArrayElements(a, 0);
    jb = env->GetDoubleArrayElements(b, 0);
  
    for(int i = 0; i < nnz; i++) {
        sa[i] = (double)ja[i];
        sasub[i] = (int)jasub[i];
    }
    for(int j = 0; j < n; j++)
        sb[j] = (double)jb[j];
    for(int k=0; k<n+1; k++)
        sxa[k] = (int)jxa[k];

    double* res1 =solve2(m, n, nnz, sa, sasub, sxa, sb, sperm_c, setree);
   
    env->SetIntArrayRegion(perm_c, 0, n, (const jint*)sperm_c);
    env->SetIntArrayRegion(etree, 0, n, (const jint*)setree);

	env->ReleaseDoubleArrayElements(a, ja, JNI_ABORT);
	env->ReleaseIntArrayElements(asub, jasub, JNI_ABORT);	
	env->ReleaseIntArrayElements(xa, jxa, JNI_ABORT);
	env->ReleaseDoubleArrayElements(b, jb, JNI_ABORT);		
	
    env->SetDoubleArrayRegion(b, 0, n, (const jdouble*)res1);	
	
	SUPERLU_FREE (res1);
	return b;
  }

/*
 * Class:     jsuperlu_superlu
 * Method:    resolve
 * Signature: (III[D[I[I[D[I[I)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_jpscpu_LinearSolver_solve3
  (JNIEnv *env, jobject obj, jint m, jint n, jint nnz, jdoubleArray a, jintArray asub, jintArray xa, jdoubleArray b, jintArray perm_c, jintArray etree){
    jint *jasub, *jxa;
    jint *jperm_c, *jetree;
    jdouble *ja, *jb;
    
	double *sa;
	int *sasub, *sxa;
	double sb[n];
	int sperm_c[n];
    int setree[n];
		
	
	if (!(sa = doubleMalloc(nnz))) ABORT("Malloc fails for a[].");
	if (!(sasub = intMalloc(nnz))) ABORT("Malloc fails for asub[].");
	if (!(sxa = intMalloc(n + 1))) ABORT("Malloc fails for xa[].");
	
    jasub = env->GetIntArrayElements(asub, 0);
    jxa = env->GetIntArrayElements(xa, 0);
    ja = env->GetDoubleArrayElements(a, 0);
    jb = env->GetDoubleArrayElements(b, 0);
    jperm_c = env->GetIntArrayElements(perm_c, 0);
    jetree = env->GetIntArrayElements(etree, 0);
	
    for(int i = 0; i < nnz; i++){
        sa[i] = (double)ja[i];
        sasub[i] = (int)jasub[i];
    }
    for(int j = 0; j < n; j++){
        sb[j] = (double)jb[j];
        sperm_c[j] = (int)jperm_c[j];
        setree[j] = (int)jetree[j];
    }
    for(int k = 0; k < n+1; k++)
        sxa[k] = (int)jxa[k];    
	
    double* res2 = solve3(m, n, nnz, sa, sasub, sxa, sb, sperm_c, setree);
    //double* res2 = doubleMalloc(n);	
	
	env->ReleaseDoubleArrayElements(a, ja, JNI_ABORT);
	env->ReleaseIntArrayElements(asub, jasub, JNI_ABORT);
	env->ReleaseIntArrayElements(xa, jxa, JNI_ABORT);
	env->ReleaseDoubleArrayElements(b, jb, JNI_ABORT);
	env->ReleaseIntArrayElements(perm_c, jperm_c, JNI_ABORT);
	env->ReleaseIntArrayElements(etree, jetree, JNI_ABORT);
	
	env->SetDoubleArrayRegion(b, 0, n, (const jdouble*)res2);	    	
	
	SUPERLU_FREE (res2);
	return b;
}

JNIEXPORT jdoubleArray JNICALL Java_jpscpu_LinearSolver_solve4
  (JNIEnv *env, jobject obj, jint m, jint n, jint nnz, jdoubleArray a, jintArray asub, jintArray xa, jdoubleArray b, jintArray perm_c, jintArray perm_r, jintArray etree, jdoubleArray R, jdoubleArray C, jobject L, jobject U,  jintArray equed_int) {
	
	jint *jasub, *jxa, *jequed;
    jdouble *ja, *jb;    
	
	double *sa;
	int *sasub, *sxa;
	double sb[n];
	
	int sperm_c[m];
    int sperm_r[n];
    int setree[n];
	double sR[m], sC[n];	
	SuperMatrix sL, sU;	   

	jasub = env->GetIntArrayElements(asub, 0);
    jxa = env->GetIntArrayElements(xa, 0);
    jequed = env->GetIntArrayElements(equed_int, 0);
    ja = env->GetDoubleArrayElements(a, 0);
    jb = env->GetDoubleArrayElements(b, 0);
	
	if (!(sa = doubleMalloc(nnz))) ABORT("Malloc fails for a[].");
	if (!(sasub = intMalloc(nnz))) ABORT("Malloc fails for asub[].");
	if (!(sxa = intMalloc(n + 1))) ABORT("Malloc fails for xa[].");
	for(int i = 0; i < nnz; i++){
        sa[i] = (double)ja[i];
        sasub[i] = (int)jasub[i];
    }
    for(int j = 0; j < n; j++)
        sb[j] = (double)jb[j];     
    for(int k = 0; k < n + 1; k++)
        sxa[k] = (int)jxa[k]; 
    double* res1 = solve4(m, n, nnz, sa, sasub, sxa, sb, sperm_c, sperm_r, setree, sR, sC, &sL, &sU, ja, jequed);
	
    env->SetIntArrayRegion(perm_c, 0, n, (const jint*)sperm_c);
    env->SetIntArrayRegion(perm_r, 0, m, (const jint*)sperm_r);
    env->SetIntArrayRegion(etree, 0, n, (const jint*)setree);
    env->SetDoubleArrayRegion(R, 0, m, (const jdouble*)sR);
    env->SetDoubleArrayRegion(C, 0, n, (const jdouble*)sC);
	
	env->ReleaseDoubleArrayElements(a, ja, JNI_ABORT);
	env->ReleaseDoubleArrayElements(b, jb, JNI_ABORT);	
	env->ReleaseIntArrayElements(asub, jasub, JNI_ABORT);	
	env->ReleaseIntArrayElements(xa, jxa, JNI_ABORT);	    	
	env->ReleaseIntArrayElements(equed_int, jequed, 0);	    	
	
	SCformat *Lstore = (SCformat *) sL.Store;
    NCformat *Ustore = (NCformat *) sU.Store;
	
	jclass clazz = env->FindClass("jpscpu/SCformat");
	jmethodID methodId = env->GetMethodID(clazz, "setNnz", "(I)V");
	env->CallVoidMethod(L, methodId, Lstore->nnz); 
	
	methodId = env->GetMethodID(clazz, "setNsuper", "(I)V");
	env->CallVoidMethod(L, methodId, Lstore->nsuper); 
	
	jdoubleArray nzval = env->NewDoubleArray(Lstore->nzval_colptr[n]);
	env->SetDoubleArrayRegion(nzval, 0, Lstore->nzval_colptr[n], (const jdouble*)Lstore->nzval);
	methodId = env->GetMethodID(clazz, "setNzval", "([D)V");
	env->CallVoidMethod(L, methodId, nzval);
	env->DeleteLocalRef(nzval);
	
	jintArray nzval_colptr = env->NewIntArray(n + 1);
	env->SetIntArrayRegion(nzval_colptr, 0, n + 1, (const jint*)Lstore->nzval_colptr);
	methodId = env->GetMethodID(clazz, "setNzval_colptr", "([I)V");
	env->CallVoidMethod(L, methodId, nzval_colptr);
	env->DeleteLocalRef(nzval_colptr);
	
	jintArray rowind = env->NewIntArray(Lstore->rowind_colptr[n]);
	env->SetIntArrayRegion(rowind, 0, Lstore->rowind_colptr[n], (const jint*)Lstore->rowind);
	methodId = env->GetMethodID(clazz, "setRowind", "([I)V");
	env->CallVoidMethod(L, methodId, rowind);
	env->DeleteLocalRef(rowind);
	
	jintArray rowind_colptr = env->NewIntArray(n + 1);
	env->SetIntArrayRegion(rowind_colptr, 0, n + 1, (const jint*)Lstore->rowind_colptr);
	methodId = env->GetMethodID(clazz, "setRowind_colptr", "([I)V");
	env->CallVoidMethod(L, methodId, rowind_colptr);
	env->DeleteLocalRef(rowind_colptr);
	
	jintArray col_to_sup = env->NewIntArray(n);
	env->SetIntArrayRegion(col_to_sup, 0, n, (const jint*)Lstore->col_to_sup);
	methodId = env->GetMethodID(clazz, "setCol_to_sup", "([I)V");
	env->CallVoidMethod(L, methodId, col_to_sup);
	env->DeleteLocalRef(col_to_sup);
	
	jintArray sup_to_col = env->NewIntArray(Lstore->nsuper + 2);
	env->SetIntArrayRegion(sup_to_col, 0, Lstore->nsuper + 2, (const jint*)Lstore->sup_to_col);
	methodId = env->GetMethodID(clazz, "setSup_to_col", "([I)V");
	env->CallVoidMethod(L, methodId, sup_to_col);
	env->DeleteLocalRef(sup_to_col);		
	
	clazz = env->FindClass("jpscpu/NCformat");
	methodId = env->GetMethodID(clazz, "setNnz", "(I)V");
	env->CallVoidMethod(U, methodId, Ustore->nnz); 		
	
	jdoubleArray nzval_U = env->NewDoubleArray(Ustore->colptr[n]);
	env->SetDoubleArrayRegion(nzval_U, 0, Ustore->colptr[n], (const jdouble*)Ustore->nzval);
	methodId = env->GetMethodID(clazz, "setNzval", "([D)V");
	env->CallVoidMethod(U, methodId, nzval_U);
	env->DeleteLocalRef(nzval_U);	
	
	jintArray rowind_U = env->NewIntArray(Ustore->colptr[n]);
	env->SetIntArrayRegion(rowind_U, 0, Ustore->colptr[n], (const jint*)Ustore->rowind);
	methodId = env->GetMethodID(clazz, "setRowind", "([I)V");
	env->CallVoidMethod(U, methodId, rowind_U);
	env->DeleteLocalRef(rowind_U);

	jintArray colptr = env->NewIntArray(n + 1);
	env->SetIntArrayRegion(colptr, 0, n + 1, (const jint*)Ustore->colptr);
	methodId = env->GetMethodID(clazz, "setColptr", "([I)V");
	env->CallVoidMethod(U, methodId, colptr);
	env->DeleteLocalRef(colptr);

	Destroy_SuperNode_Matrix(&sL);
    Destroy_CompCol_Matrix(&sU);	
	
	env->SetDoubleArrayRegion(b, 0, n, (const jdouble*)res1);	
	SUPERLU_FREE (res1);	

	return b;
}

/*
 * Class:     jpscpu_LinearSolver
 * Method:    solve5
 * Signature: (III[D[I[I[D[I[I[I[D[DLjpscpu/SCformat;Ljpscpu/NCformat;)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_jpscpu_LinearSolver_solve5
 (JNIEnv *env, jobject obj, jint m, jint n, jint nnz, jdoubleArray a, jintArray asub, jintArray xa, 
 jdoubleArray b, jintArray perm_c, jintArray perm_r, jintArray etree, jdoubleArray R, jdoubleArray C,  
 jint nnz_L, jint nsuper_L, jdoubleArray nzval_L, jintArray nzval_colptr_L, jintArray rowind_L, 
 jintArray rowind_colptr_L, jintArray col_to_sup_L, jintArray sup_to_col_L, 
 jint nnz_U, jdoubleArray nzval_U, jintArray rowind_U, jintArray colptr_U, jintArray equed_int) {
	
	jint *jasub, *jxa, *jperm_c, *jperm_r, *jetree, *jequed;	
	jint *jnzval_colptr_L, *jrowind_L, *jrowind_colptr_L, *jcol_to_sup_L, *jsup_to_col_L, *jrowind_U, *jcolptr_U;
	jdouble *ja, *jb, *jR, *jC, *jnzval_L, *jnzval_U;
	
    double sb[n], sR[m], sC[n];
	int sperm_r[m],sperm_c[n],setree[n];	
	int *snzval_colptr_L, *srowind_L, *srowind_colptr_L, *scol_to_sup_L, *ssup_to_col_L, *srowind_U, *scolptr_U;
	double *sa, *snzval_L, *snzval_U;
	int *sasub, *sxa;	
	
	ja = env->GetDoubleArrayElements(a, 0);
    jb = env->GetDoubleArrayElements(b, 0);
    jR = env->GetDoubleArrayElements(R, 0);
    jC = env->GetDoubleArrayElements(C, 0);
    jnzval_L = env->GetDoubleArrayElements(nzval_L, 0);
    jnzval_U = env->GetDoubleArrayElements(nzval_U, 0);
    
	jasub = env->GetIntArrayElements(asub, 0);
    jxa = env->GetIntArrayElements(xa, 0);
	jperm_c = env->GetIntArrayElements(perm_c, 0);
    jperm_r = env->GetIntArrayElements(perm_r, 0);
    jetree = env->GetIntArrayElements(etree, 0);
    jequed = env->GetIntArrayElements(equed_int, 0);
    
	jnzval_colptr_L = env->GetIntArrayElements(nzval_colptr_L, 0);
	jrowind_L = env->GetIntArrayElements(rowind_L, 0);
	jrowind_colptr_L = env->GetIntArrayElements(rowind_colptr_L, 0);
	jcol_to_sup_L = env->GetIntArrayElements(col_to_sup_L, 0);
	jsup_to_col_L = env->GetIntArrayElements(sup_to_col_L, 0);
	
	jrowind_U = env->GetIntArrayElements(rowind_U, 0);
	jcolptr_U = env->GetIntArrayElements(colptr_U, 0);

	if (!(sa = doubleMalloc(nnz))) ABORT("Malloc fails for a[].");
	if (!(sasub = intMalloc(nnz))) ABORT("Malloc fails for asub[].");
	if (!(sxa = intMalloc(n + 1))) ABORT("Malloc fails for xa[].");		
	
	if (!(snzval_L = doubleMalloc(jnzval_colptr_L[n]))) ABORT("Malloc fails for nzval_L[].");
	if (!(snzval_U = doubleMalloc(jcolptr_U[n]))) ABORT("Malloc fails for nzval_U[].");
	
	if (!(snzval_colptr_L = intMalloc(n + 1))) ABORT("Malloc fails for nzval_colptr_L[].");
	if (!(srowind_L = intMalloc(jrowind_colptr_L[n]))) ABORT("Malloc fails for rowind_L[].");	
	if (!(srowind_colptr_L = intMalloc(n + 1))) ABORT("Malloc fails for rowind_colptr_L[].");	
	if (!(scol_to_sup_L = intMalloc(n + 1))) ABORT("Malloc fails for col_to_sup_L[].");	
	if (!(ssup_to_col_L = intMalloc(n + 1))) ABORT("Malloc fails for sup_to_col_L[].");	
	
	if (!(srowind_U = intMalloc(jcolptr_U[n]))) ABORT("Malloc fails for rowind_U[].");	
	if (!(scolptr_U = intMalloc(n + 1))) ABORT("Malloc fails for colptr_U[].");	
    
	for(int i = 0; i < nnz; i++){
        sa[i] = (double)ja[i];
        sasub[i] = (int)jasub[i];		
    }	

    for(int i = 0; i < n; i++) {
        sb[i] = (double)jb[i];
        sperm_c[i] = (int)jperm_c[i];
        setree[i] = (int)jetree[i];
		sC[i] = (double)jC[i];
		scol_to_sup_L[i] = (int)jcol_to_sup_L[i];
    }	
	for(int i = 0; i < m; i++) {
		sperm_r[i] = (int)jperm_r[i];
		sR[i] = (double)jR[i];
	}	

    for(int i = 0; i < n + 1; i++) {
        sxa[i] = (int)jxa[i]; 
		
		snzval_colptr_L[i] = (int)jnzval_colptr_L[i];
		srowind_colptr_L[i] = (int)jrowind_colptr_L[i];
		scolptr_U[i] = (int)jcolptr_U[i];	
	}
	for(int i = 0; i < nsuper_L + 2; i++)	
		ssup_to_col_L[i] = (int)jsup_to_col_L[i];
	for(int i = 0; i < snzval_colptr_L[n]; i++)
		 snzval_L[i] = (double)jnzval_L[i];
	for(int i = 0; i < srowind_colptr_L[n]; i++)       
        srowind_L[i] = (int)jrowind_L[i];		
    
	for(int i = 0; i < scolptr_U[n]; i++){
        snzval_U[i] = (double)jnzval_U[i];
        srowind_U[i] = (int)jrowind_U[i];		
    }	
	
	SuperMatrix sL, sU;
	SCformat Lstore;
    NCformat Ustore;
	Lstore.nnz = (int)nnz_L;
	Lstore.nsuper = (int)nsuper_L;
	Lstore.nzval = snzval_L;
	Lstore.nzval_colptr = snzval_colptr_L;
	Lstore.rowind = srowind_L; 
	Lstore.rowind_colptr = srowind_colptr_L;
	Lstore.col_to_sup = scol_to_sup_L;
	Lstore.sup_to_col = ssup_to_col_L;
	sL.Stype = SLU_SC;
	sL.Dtype = SLU_D;
	sL.Mtype = SLU_TRLU;
	sL.nrow = m;
	sL.ncol = n;
	sL.Store = &Lstore;	
	
	Ustore.nnz = (int)nnz_U;
	Ustore.nzval = snzval_U;
	Ustore.rowind = srowind_U; 
	Ustore.colptr = scolptr_U;	
	sU.Stype = SLU_NC;
	sU.Dtype = SLU_D;
	sU.Mtype = SLU_TRU;
	sU.nrow = m;
	sU.ncol = n;
	sU.Store = &Ustore;	

	
	double* res2 = solve5(m, n, nnz, sa, sasub, sxa, sb, sperm_c, sperm_r, setree, sR, sC, &sL, &sU, jequed); 	
	
	env->ReleaseDoubleArrayElements(a, ja, JNI_ABORT);
	env->ReleaseDoubleArrayElements(b, jb, JNI_ABORT);
	env->ReleaseDoubleArrayElements(R, jR, JNI_ABORT);
	env->ReleaseDoubleArrayElements(C, jC, JNI_ABORT);
	env->ReleaseDoubleArrayElements(nzval_L, jnzval_L, JNI_ABORT);
	env->ReleaseDoubleArrayElements(nzval_U, jnzval_U, JNI_ABORT);
	
	env->ReleaseIntArrayElements(asub, jasub, JNI_ABORT);
	env->ReleaseIntArrayElements(xa, jxa, JNI_ABORT);
	env->ReleaseIntArrayElements(perm_c, jperm_c, JNI_ABORT);
	env->ReleaseIntArrayElements(perm_r, jperm_r, JNI_ABORT);
	env->ReleaseIntArrayElements(etree, jetree, JNI_ABORT);
	
	env->ReleaseIntArrayElements(nzval_colptr_L, jnzval_colptr_L, JNI_ABORT);    
	env->ReleaseIntArrayElements(rowind_L, jrowind_L, JNI_ABORT);    
	env->ReleaseIntArrayElements(rowind_colptr_L, jrowind_colptr_L, JNI_ABORT);    
	env->ReleaseIntArrayElements(col_to_sup_L, jcol_to_sup_L, JNI_ABORT);    
	env->ReleaseIntArrayElements(sup_to_col_L, jsup_to_col_L, JNI_ABORT);    

	env->ReleaseIntArrayElements(rowind_U, jrowind_U, JNI_ABORT);    
	env->ReleaseIntArrayElements(colptr_U, jcolptr_U, JNI_ABORT);    
	env->ReleaseIntArrayElements(equed_int, jequed, JNI_ABORT);    
	
	env->SetDoubleArrayRegion(b, 0, n, (const jdouble*)res2);	
	SUPERLU_FREE (res2);
	
	//Destroy_SuperNode_Matrix(&sL);
	SUPERLU_FREE ( Lstore.rowind );
    SUPERLU_FREE ( Lstore.rowind_colptr );
    SUPERLU_FREE ( Lstore.nzval );
    SUPERLU_FREE ( Lstore.nzval_colptr );
    SUPERLU_FREE ( Lstore.col_to_sup );
    SUPERLU_FREE ( Lstore.sup_to_col );
    //SUPERLU_FREE ( Lstore );
	//Destroy_CompCol_Matrix(&sU);	
	SUPERLU_FREE( Ustore.rowind );
    SUPERLU_FREE( Ustore.colptr );
    SUPERLU_FREE( Ustore.nzval );
	
    return b;
}

//====================== main fro test ==============
int main(int argc, char *argv[]) {
	/*
	* Purpose
	* =======
	*
	* This is the small 5x5 example used in the Sections 2 and 3 of the
	* Users’ Guide to illustrate how to call a SuperLU routine, and the
	* matrix data structures used by SuperLU.
	*
	*/
	SuperMatrix A, L, U, B;
	double *a, *rhs;
	double s, u, p, e, r, l;
	int *asub, *xa;
	int *perm_r; /* row permutations from partial pivoting */
	
	int *perm_c; /* column permutation vector */
	int nrhs, info, i, m, n, nnz, permc_spec;
	superlu_options_t options;
	SuperLUStat_t stat;
	/* Initialize matrix A. */
	m = n = 5;
	nnz = 12;
	if ( !(a = doubleMalloc(nnz)) ) ABORT("Malloc fails for a[].");
	if ( !(asub = intMalloc(nnz)) ) ABORT("Malloc fails for asub[].");
	if ( !(xa = intMalloc(n+1)) ) ABORT("Malloc fails for xa[].");
	s = 19.0; u = 21.0; p = 16.0; e = 5.0; r = 18.0; l = 12.0;
	a[0] = s; a[1] = l; a[2] = l; a[3] = u; a[4] = l; a[5] = l;
	a[6] = u; a[7] = p; a[8] = u; a[9] = e; a[10]= u; a[11]= r;
	asub[0] = 0; asub[1] = 1; asub[2] = 4; asub[3] = 1;
	asub[4] = 2; asub[5] = 4; asub[6] = 0; asub[7] = 2;
	asub[8] = 0; asub[9] = 3; asub[10]= 3; asub[11]= 4;
	xa[0] = 0; xa[1] = 3; xa[2] = 6; xa[3] = 8; xa[4] = 10; xa[5] = 12;
	/* Create matrix A in the format expected by SuperLU. */
	dCreate_CompCol_Matrix(&A, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);
	/* Create right-hand side matrix B. */
	nrhs = 1;
	if ( !(rhs = doubleMalloc(m * nrhs)) ) ABORT("Malloc fails for rhs[].");
	for (i = 0; i < m; ++i) rhs[i] = 1.0;
	dCreate_Dense_Matrix(&B, m, nrhs, rhs, m, SLU_DN, SLU_D, SLU_GE);
	if ( !(perm_r = intMalloc(m)) ) ABORT("Malloc fails for perm_r[].");
	if ( !(perm_c = intMalloc(n)) ) ABORT("Malloc fails for perm_c[].");
	/* Set the default input options. */
	set_default_options(&options);
	options.ColPerm = NATURAL;
	/* Initialize the statistics variables. */
	StatInit(&stat);
	/* Solve the linear system. */
	dgssv(&options, &A, perm_c, perm_r, &L, &U, &B, &stat, &info);
	dPrint_CompCol_Matrix("A", &A);
	dPrint_CompCol_Matrix("U", &U);
	dPrint_SuperNode_Matrix("L", &L);
	print_int_vec("\nperm_r", m, perm_r);
	
	/* De-allocate storage */
	SUPERLU_FREE (rhs);
	SUPERLU_FREE (perm_r);
	SUPERLU_FREE (perm_c);
	Destroy_CompCol_Matrix(&A);
	Destroy_SuperMatrix_Store(&B);
	Destroy_SuperNode_Matrix(&L);
	Destroy_CompCol_Matrix(&U);
	StatFree(&stat);
	return 1;
}

