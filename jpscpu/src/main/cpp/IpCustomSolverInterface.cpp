// Copyright (C) 2006, 2012 Damien Hocking, KBC Advanced Technologies
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors: Damien Hocking                 KBC    2006-03-20
//        (included his original contribution into Ipopt package on 2006-03-25)
//          Andreas Waechter               IBM    2006-03-25
//           (minor changes and corrections)
//          Scott Turnberg                 CMU    2006-05-12
//           (major revision)
//           (incorporated by AW on 2006-11-11 into Ipopt package)

// The following line is a fix for otherwise twice-defined global variable
// (This would have to be taken out for a parallel MUMPS version!)
#define MPI_COMM_WORLD IPOPT_MPI_COMM_WORLD
// The first header to include is the one for MPI.  
// In newer ThirdParty/Mumps, mpi.h is renamed to mumps_mpi.h.
// We get informed about this by having COIN_USE_MUMPS_MPI_H defined.
#ifdef COIN_USE_MUMPS_MPI_H
#include "mpi.h"
#else
#include "mpi.h"
#endif

#include "IpCustomSolverInterface.hpp"
#include "jpscpu_LinearSolver.h"

#include "slu_ddefs.h"
#include "sstream"

#include <algorithm>
#include <vector>
#include <iterator>  


#include "dmumps_c.h"

//#ifdef HAVE_CMATH
# include <cmath>
//#else
//# ifdef HAVE_MATH_H
//#  include <math.h>
//# else
//#  error "don't have header file for math"
//# endif
//#endif
//
//#ifdef HAVE_CSTDLIB
# include <cstdlib>
#include <ClpCholeskyMumps.hpp>
//#else
//# ifdef HAVE_STDLIB_H
//#  include <stdlib.h>
//# else
//#  error "don't have header file for stdlib"
//# endif
//#endif

using namespace std; 

namespace Ipopt
{
#if COIN_IPOPT_VERBOSITY > 0
  static const Index dbg_verbosity = 0;
#endif

#define USE_COMM_WORLD -987654

  int CustomSolverInterface::instancecount_mpi = 0;

  CustomSolverInterface::CustomSolverInterface()
  {
    DBG_START_METH("CustomSolverInterface::CustomSolverInterface()",
                   dbg_verbosity);
    //initialize mumps
    call_counter = 0;
    DMUMPS_STRUC_C *mumps_ = new DMUMPS_STRUC_C;
#ifndef MUMPS_MPI_H
#if defined(HAVE_MPI_INITIALIZED)
    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if( !mpi_initialized )
    {
       int argc = 1;
       char** argv = NULL;
       MPI_Init(&argc, &argv);
       assert(instancecount_mpi == 0);
       instancecount_mpi = 1;
    }
    else if( instancecount_mpi > 0 )
       ++instancecount_mpi;
#endif
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
#endif
    mumps_->n = 0;
    mumps_->nz = 0;
    mumps_->a = NULL;
    mumps_->jcn = NULL;
    mumps_->irn = NULL;
    mumps_->job = -1;//initialize mumps
    mumps_->par = 1;//working host for sequential version
    mumps_->sym = 2;//general symetric matrix
    mumps_->comm_fortran = USE_COMM_WORLD;
    dmumps_c(mumps_);
    mumps_->icntl[1] = 0;
    mumps_->icntl[2] = 0;//QUIETLY!
    mumps_->icntl[3] = 0;
    mumps_ptr_ = (void*)mumps_;
    // call_counter = 0；
  }

  CustomSolverInterface::~CustomSolverInterface()
  {
    DBG_START_METH("CustomSolverInterface::~CustomSolverInterface()",
                   dbg_verbosity);

    DMUMPS_STRUC_C* mumps_ = (DMUMPS_STRUC_C*)mumps_ptr_;
    mumps_->job = -2; //terminate mumps
    dmumps_c(mumps_);
#ifndef MUMPS_MPI_H
#ifdef HAVE_MPI_INITIALIZED
    if( instancecount_mpi == 1 )
    {
       int mpi_finalized;
       MPI_Finalized(&mpi_finalized);
       assert(!mpi_finalized);
       MPI_Finalize();
    }
    --instancecount_mpi;
#endif
#endif
    delete [] mumps_->a;
    delete mumps_;
  }

  void CustomSolverInterface::RegisterOptions(SmartPtr<RegisteredOptions> roptions)
  {
    roptions->AddBoundedNumberOption(
      "mumps_pivtol",
      "Pivot tolerance for the linear solver MUMPS.",
      0, false, 1, false, 1e-6,
      "A smaller number pivots for sparsity, a larger number pivots for "
      "stability.  This option is only available if Ipopt has been compiled "
      "with MUMPS.");
    roptions->AddBoundedNumberOption(
      "mumps_pivtolmax",
      "Maximum pivot tolerance for the linear solver MUMPS.",
      0, false, 1, false, 0.1,
      "Ipopt may increase pivtol as high as pivtolmax to get a more accurate "
      "solution to the linear system.  This option is only available if "
      "Ipopt has been compiled with MUMPS.");
    roptions->AddLowerBoundedIntegerOption(
      "mumps_mem_percent",
      "Percentage increase in the estimated working space for MUMPS.",
      0, 1000,
      "In MUMPS when significant extra fill-in is caused by numerical "
      "pivoting, larger values of mumps_mem_percent may help use the "
      "workspace more efficiently.  On the other hand, if memory requirement "
      "are too large at the very beginning of the optimization, choosing a "
      "much smaller value for this option, such as 5, might reduce memory "
      "requirements.");
    roptions->AddBoundedIntegerOption(
      "mumps_permuting_scaling",
      "Controls permuting and scaling in MUMPS",
      0, 7, 7,
      "This is ICNTL(6) in MUMPS.");
    roptions->AddBoundedIntegerOption(
      "mumps_pivot_order",
      "Controls pivot order in MUMPS",
      0, 7, 7,
      "This is ICNTL(7) in MUMPS.");
    roptions->AddBoundedIntegerOption(
      "mumps_scaling",
      "Controls scaling in MUMPS",
      -2, 77, 77,
      "This is ICNTL(8) in MUMPS.");
    roptions->AddNumberOption(
      "mumps_dep_tol",
      "Pivot threshold for detection of linearly dependent constraints in MUMPS.",
      0.0,
      "When MUMPS is used to determine linearly dependent constraints, this "
      "is determines the threshold for a pivot to be considered zero.  This "
      "is CNTL(3) in MUMPS.");
  }

  bool CustomSolverInterface::InitializeImpl(const OptionsList& options,
      const std::string& prefix)
  {
    options.GetNumericValue("mumps_pivtol", pivtol_, prefix);
    if (options.GetNumericValue("mumps_pivtolmax", pivtolmax_, prefix)) {
      ASSERT_EXCEPTION(pivtolmax_>=pivtol_, OPTION_INVALID,
                       "Option \"mumps_pivtolmax\": This value must be between "
                       "mumps_pivtol and 1.");
    }
    else {
      pivtolmax_ = Max(pivtolmax_, pivtol_);
    }

    options.GetIntegerValue("mumps_mem_percent",
                            mem_percent_, prefix);

    // The following option is registered by OrigIpoptNLP
    options.GetBoolValue("warm_start_same_structure",
                         warm_start_same_structure_, prefix);

    options.GetIntegerValue("mumps_permuting_scaling",
                            mumps_permuting_scaling_, prefix);
    options.GetIntegerValue("mumps_pivot_order", mumps_pivot_order_, prefix);
    options.GetIntegerValue("mumps_scaling", mumps_scaling_, prefix);
    options.GetNumericValue("mumps_dep_tol", mumps_dep_tol_, prefix);

    // Reset all private data
    initialized_ = false;
    pivtol_changed_ = false;
    refactorize_ = false;
    have_symbolic_factorization_ = false;

    DMUMPS_STRUC_C* mumps_ = (DMUMPS_STRUC_C*)mumps_ptr_;
    if (!warm_start_same_structure_) {
      mumps_->n = 0;
      mumps_->nz = 0;
    }
    else {
      ASSERT_EXCEPTION(mumps_->n>0 && mumps_->nz>0, INVALID_WARMSTART,
                       "CustomSolverInterface called with warm_start_same_structure, but the problem is solved for the first time.");
    }

    return true;
  }

  ESymSolverStatus CustomSolverInterface::MultiSolve(bool new_matrix,
                                                     const Index *ia,
                                                     const Index *ja,
                                                     Index nrhs,
                                                     double *rhs_vals,
                                                     bool check_NegEVals,
                                                     Index numberOfNegEVals)
  {
      DBG_START_METH("MumpsSolverInterface::MultiSolve", dbg_verbosity);
      DBG_ASSERT(!check_NegEVals || ProvidesInertia());
      DBG_ASSERT(initialized_);
      DBG_ASSERT(((DMUMPS_STRUC_C *)mumps_ptr_)->irn == ia);
      DBG_ASSERT(((DMUMPS_STRUC_C *)mumps_ptr_)->jcn == ja);

      DMUMPS_STRUC_C *mumps_ = (DMUMPS_STRUC_C *)mumps_ptr_;
      int m, n, nnz;

      // cout << new_matrix << endl;
      
      m = mumps_->n;
      n = mumps_->n;
      nnz = mumps_->nz;

      std::list<int> irn(ia, ia + nnz);
      std::list<int> jcn(ja, ja + nnz);
      std::list<double> a(mumps_->a, mumps_->a + nnz);

      std::list<int>::iterator iter_irn_i = irn.begin();
      std::list<int>::iterator iter_jcn_i = jcn.begin();
      std::list<double>::iterator iter_a_i = a.begin();
      
      std::list<int>::iterator iter_irn_j;
      std::list<int>::iterator iter_jcn_j;
      std::list<double>::iterator iter_a_j;

      while ((iter_irn_i != irn.end()) && (iter_jcn_i != jcn.end()))
      {
        // 让j指向i下一个元素
        iter_irn_j = iter_irn_i;
        iter_jcn_j = iter_jcn_i;
        iter_a_j = iter_a_i;
        iter_irn_j++;
        iter_jcn_j++;
        iter_a_j++;

        //j从i+1处开始循环
        while ((iter_irn_j != irn.end()) && (iter_jcn_j != jcn.end()))
        {
          if ((*iter_irn_i == *iter_irn_j) && (*iter_jcn_i == *iter_jcn_j)) //如果三元组中有重复元素
          {
            *iter_a_i += *iter_a_j; //在a中第i个位置加上j位置的重复元素

            //擦除j位置的数据，并让迭代器指向下一个数
            irn.erase(iter_irn_j++);
            jcn.erase(iter_jcn_j++);
            a.erase(iter_a_j++);
          }
          else
          {
            iter_irn_j++;
            iter_jcn_j++;
            iter_a_j++;
          }
        }

        //打印irn、jcn、和a中的所有元素
        // cout << *iter_irn_i << " " << *iter_jcn_i << " " << *iter_a_i << endl;

        if (*iter_irn_i != *iter_jcn_i)
        {
          irn.push_front(*iter_jcn_i);
          jcn.push_front(*iter_irn_i);
          a.push_front(*iter_a_i);
        }

        iter_irn_i++;
        iter_jcn_i++;
        iter_a_i++;
      }

      // cout << "三元组大小：" << endl << irn.size() << " " << jcn.size() << " " << a.size() << endl;
      nnz = irn.size();
      double a_unique[nnz];
      int irn_unique[nnz], jcn_unique[nnz];

      iter_irn_i = irn.begin();
      iter_jcn_i = jcn.begin();
      iter_a_i = a.begin();    

      //list转为数组
      int temp_count = 0;
      while ((iter_irn_i != irn.end()) && (iter_jcn_i != jcn.end()))
      {
        irn_unique[temp_count] = *iter_irn_i-1;
        jcn_unique[temp_count] = *iter_jcn_i-1;
        a_unique[temp_count] = *iter_a_i;

        temp_count++;
        iter_irn_i++;
        iter_jcn_i++;
        iter_a_i++;
      }

      //先按列进行排序
      Qsort(jcn_unique, irn_unique, a_unique, 0, sizeof(jcn_unique) / sizeof(int) - 1);
      
      //再按行进行排序
      int last_column = 0;
      for (int this_column = 1; this_column < nnz; this_column++)
      {
        if (jcn_unique[this_column] != jcn_unique[last_column])
        {
          Qsort(irn_unique, jcn_unique, a_unique, last_column, this_column - 1);
          last_column = this_column;
        }
      }

      //现在三元组已经按列从小到大进行排列，相同列的元素其行也按从小到大排列。下面进行CSC格式转换。
      //asub代表该元素所处的行坐标，xa代表该元素在csc_a数组中所处的位置。

      // cout << m << " " << n << " " << nnz << endl;

      //这里在申请内存空间的时候不用自己释放，superlu会destroy。
      double *csc_a;
      int *csc_asub, *csc_xa;
      int asub_index = 0;
    
      // double csc_a[nnz];
      // int csc_asub[nnz], csc_xa[n+1];

      csc_a = new double[nnz];
      csc_asub = new int[nnz];
      csc_xa = new int[n+1];

      // if ( !(csc_a = doubleMalloc(nnz)) ) ABORT("Malloc fails for a[].");
      // if ( !(csc_asub = intMalloc(nnz)) ) ABORT("Malloc fails for asub[].");
      // if ( !(csc_xa = intMalloc(n+1)) ) ABORT("Malloc fails for xa[].");

      csc_a[0] = a_unique[0];
      csc_asub[0] = irn_unique[0];
      csc_xa[0] = asub_index;
      asub_index++;
      for (int i = 1; i < nnz; i++)
      {
        csc_a[i] = a_unique[i];
        csc_asub[i] = irn_unique[i];
        if (jcn_unique[i] > jcn_unique[i - 1])
        {
          csc_xa[asub_index] = i;
          asub_index++;
        }
      }
      csc_xa[asub_index] = nnz;

      //调用superlu求解器开始     

      // int *perm_c, *etree;
      // perm_c = new int[n];
      // etree = new int[n];

      // if ( !(perm_c = intMalloc(n)) ) ABORT("Malloc fails for perm_c[].");
      // if ( !(etree = intMalloc(n)) ) ABORT("Malloc fails for etree[].");

      // if (pivtol_changed_)
      // {
      //   cout << "pivtol_changed_" << endl;
      //   cout << pivtol_changed_ << endl;
      //   DBG_PRINT((1, "Pivot tolerance has changed.\n"));
      //   pivtol_changed_ = false;
      //   // If the pivot tolerance has been changed but the matrix is not
      //   // new, we have to request the values for the matrix again to do
      //   // the factorization again.
      //   if (!new_matrix)
      //   {
      //     cout << "!new_matrix" << endl;
      //     cout << !new_matrix << endl;
      //     DBG_PRINT((1, "Ask caller to call again.\n"));
      //     refactorize_ = true;
      //     return SYMSOLVER_CALL_AGAIN;
      //   }
      // }

      // // check if a factorization has to be done
      // DBG_PRINT((1, "new_matrix = %d\n", new_matrix));
      // if (new_matrix || refactorize_)
      // {
      //   cout << "!new_matrix" << endl;
      //   cout << !new_matrix << endl;
      //   ESymSolverStatus retval;
      //   // Do the symbolic facotrization if it hasn't been done yet
      //   if (!have_symbolic_factorization_)
      //   {
      //     retval = SymbolicFactorization();
      //     if (retval != SYMSOLVER_SUCCESS)
      //     {
      //       return retval;
      //     }
      //     have_symbolic_factorization_ = true;
      //   }
      //   // perform the factorization
      //   retval = Factorization(check_NegEVals, numberOfNegEVals);
      //   if (retval != SYMSOLVER_SUCCESS)
      //   {
      //     DBG_PRINT((1, "FACTORIZATION FAILED!\n"));
      //     return retval; // Matrix singular or error occurred
      //   }
      //   refactorize_ = false;
      // }

      ESymSolverStatus retval = SYMSOLVER_SUCCESS;

      if (call_counter == 0)
      {
        perm_c = new int[n];
        etree = new int[n];
        Solve2(m, n, nnz, csc_a, csc_asub, csc_xa, rhs_vals, perm_c, etree);
        call_counter++;
        return retval;
      }

      //do the solve
      //这里调用superlu的求解器，会改变rhs_vals指向的变量的值，从而获得X向量，存放在rhs_vals指针中

      if (new_matrix)
      {
        delete[] perm_c;
        delete[] etree;
        perm_c = new int[n];
        etree = new int[n];
        Solve2(m, n, nnz, csc_a, csc_asub, csc_xa, rhs_vals, perm_c, etree);
      }
      else
      {
        Solve3(m, n, nnz, csc_a, csc_asub, csc_xa, rhs_vals, perm_c, etree);
      }

      // return Solve(nrhs, rhs_vals, m, n, nnz, csc_a, csc_asub, csc_xa, perm_c, etree);
      call_counter++;
      return retval;
  }

  void CustomSolverInterface::Qsort(int *sort_array, int *position, double *a, int low, int high)
  {
      if (low >= high)
      {
          return;
      }
      int first = low;
      int last = high;
      int key = sort_array[first]; /*用字表的第一个记录作为枢轴*/
      int key_position = position[first];
      double key_a = a[first];

      while (first < last)
      {
          while (first < last && sort_array[last] >= key)
          {
              --last;
          }

          /*将比第一个小的移到低端*/
          sort_array[first] = sort_array[last];
          position[first] = position[last];
          a[first] = a[last];

          while (first < last && sort_array[first] <= key)
          {
              ++first;
          }

          /*将比第一个大的移到高端*/
          sort_array[last] = sort_array[first];
          position[last] = position[first];
          a[last] = a[first];
      }
      sort_array[first] = key; /*枢轴记录到位*/
      position[first] = key_position;
      a[first] = key_a;
      Qsort(sort_array, position, a, low, first - 1);
      Qsort(sort_array, position, a, first + 1, high);
  }

  double* CustomSolverInterface::GetValuesArrayPtr()
  {
    DMUMPS_STRUC_C* mumps_ = (DMUMPS_STRUC_C*)mumps_ptr_;
    DBG_START_METH("CustomSolverInterface::GetValuesArrayPtr",dbg_verbosity)
    DBG_ASSERT(initialized_);
    return mumps_->a;
  }

  void dump_matrix(DMUMPS_STRUC_C* mumps_data)
  {
#ifdef write_matrices
    // Dump the matrix    
    for (int i=0; i<40; i++) {
      printf("%d\n", mumps_data->icntl[i]);
    }
    
    for (int i=0; i<5; i++) {
      printf("%25.15e\n", mumps_data->cntl[i]);
    }
    printf("%-15d :N\n",mumps_data->n);
    printf("%-15d :NZ", mumps_data->nz);
    for (int i=0; i<mumps_data->nz; i++) {
      printf("\n%d %d %25.15e", mumps_data->irn[i], mumps_data->jcn[i], mumps_data->a[i]);
    }
    printf("       :values");
    // Dummy RHS for now
    for (int i=0; i<mumps_data->n; i++) {
      printf("\n%25.15e", 0.);
    }
    printf("    :RHS\n");
#endif
  }

  /* Initialize the local copy of the positions of the nonzero
      elements */
  ESymSolverStatus CustomSolverInterface::InitializeStructure(Index dim,
      Index nonzeros,
      const Index* ia,
      const Index* ja)
  {
    DMUMPS_STRUC_C* mumps_ = (DMUMPS_STRUC_C*)mumps_ptr_;
    DBG_START_METH("CustomSolverInterface::InitializeStructure", dbg_verbosity);

    ESymSolverStatus retval = SYMSOLVER_SUCCESS;
    if (!warm_start_same_structure_) {
      mumps_->n = dim;
      mumps_->nz = nonzeros;
      delete [] mumps_->a;
      mumps_->a = NULL;

      mumps_->a = new double[nonzeros];
      mumps_->irn = const_cast<int*>(ia);
      mumps_->jcn = const_cast<int*>(ja);

      // make sure we do the symbolic factorization before a real
      // factorization
      have_symbolic_factorization_ = false;
    }
    else {
      ASSERT_EXCEPTION(mumps_->n==dim && mumps_->nz==nonzeros,
                       INVALID_WARMSTART,"CustomSolverInterface called with warm_start_same_structure, but the problem size has changed.");
    }

    initialized_ = true;
    return retval;
  }


  ESymSolverStatus CustomSolverInterface::SymbolicFactorization()
  {
    DBG_START_METH("CustomSolverInterface::SymbolicFactorization",
                   dbg_verbosity);
    DMUMPS_STRUC_C* mumps_data = (DMUMPS_STRUC_C*)mumps_ptr_;

    if (HaveIpData()) {
      IpData().TimingStats().LinearSystemSymbolicFactorization().Start();
    }

    mumps_data->job = 1;//symbolic ordering pass

    //mumps_data->icntl[1] = 6;
    //mumps_data->icntl[2] = 6;//QUIETLY!
    //mumps_data->icntl[3] = 4;

    mumps_data->icntl[5] = mumps_permuting_scaling_;
    mumps_data->icntl[6] = mumps_pivot_order_;
    mumps_data->icntl[7] = mumps_scaling_;
    mumps_data->icntl[9] = 0;//no iterative refinement iterations


    mumps_data->icntl[12] = 1;//avoid lapack bug, ensures proper inertia
    mumps_data->icntl[13] = mem_percent_; //% memory to allocate over expected
    mumps_data->cntl[0] = pivtol_;  // Set pivot tolerance

    dump_matrix(mumps_data);

    Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                   "Calling MUMPS-1 for symbolic factorization at cpu time %10.3f (wall %10.3f).\n", CpuTime(), WallclockTime());
    dmumps_c(mumps_data);
    Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                   "Done with MUMPS-1 for symbolic factorization at cpu time %10.3f (wall %10.3f).\n", CpuTime(), WallclockTime());
    int error = mumps_data->info[0];
    const int& mumps_permuting_scaling_used = mumps_data->infog[22];
    const int& mumps_pivot_order_used = mumps_data->infog[6];
    Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                   "MUMPS used permuting_scaling %d and pivot_order %d.\n",
                   mumps_permuting_scaling_used, mumps_pivot_order_used);
    Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                   "           scaling will be %d.\n",
                   mumps_data->icntl[7]);

    if (HaveIpData()) {
      IpData().TimingStats().LinearSystemSymbolicFactorization().End();
    }

    //return appropriat value
    if (error == -6) {//system is singular
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                     "MUMPS returned INFO(1) = %d matrix is singular.\n",error);
      return SYMSOLVER_SINGULAR;
    }
    if (error < 0) {
      Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
                     "Error=%d returned from MUMPS in Factorization.\n",
                     error);
      return SYMSOLVER_FATAL_ERROR;
    }

    return SYMSOLVER_SUCCESS;
  }

  ESymSolverStatus CustomSolverInterface::Factorization(
    bool check_NegEVals, Index numberOfNegEVals)
  {
    DBG_START_METH("CustomSolverInterface::Factorization", dbg_verbosity);
    DMUMPS_STRUC_C* mumps_data = (DMUMPS_STRUC_C*)mumps_ptr_;

    mumps_data->job = 2;//numerical factorization

    dump_matrix(mumps_data);
    Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                   "Calling MUMPS-2 for numerical factorization at cpu time %10.3f (wall %10.3f).\n", CpuTime(), WallclockTime());
    dmumps_c(mumps_data);
    Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                   "Done with MUMPS-2 for numerical factorization at cpu time %10.3f (wall %10.3f).\n", CpuTime(), WallclockTime());
    int error = mumps_data->info[0];

    //Check for errors
    if (error == -8 || error == -9) {//not enough memory
      const Index trycount_max = 20;
      for (int trycount=0; trycount<trycount_max; trycount++) {
        Jnlst().Printf(J_WARNING, J_LINEAR_ALGEBRA,
                       "MUMPS returned INFO(1) = %d and requires more memory, reallocating.  Attempt %d\n",
                       error,trycount+1);
        Jnlst().Printf(J_WARNING, J_LINEAR_ALGEBRA,
                       "  Increasing icntl[13] from %d to ", mumps_data->icntl[13]);
        double mem_percent = mumps_data->icntl[13];
        mumps_data->icntl[13] = (Index)(2.0 * mem_percent);
        Jnlst().Printf(J_WARNING, J_LINEAR_ALGEBRA, "%d.\n", mumps_data->icntl[13]);

        dump_matrix(mumps_data);
        Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                       "Calling MUMPS-2 (repeated) for numerical factorization at cpu time %10.3f (wall %10.3f).\n", CpuTime(), WallclockTime());
        dmumps_c(mumps_data);
        Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                       "Done with MUMPS-2 (repeated) for numerical factorization at cpu time %10.3f (wall %10.3f).\n", CpuTime(), WallclockTime());
        error = mumps_data->info[0];
        if (error != -8 && error != -9)
          break;
      }
      if (error == -8 || error == -9) {
        Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
                       "MUMPS was not able to obtain enough memory.\n");
        return SYMSOLVER_FATAL_ERROR;
      }
    }

    Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                   "Number of doubles for MUMPS to hold factorization (INFO(9)) = %d\n",
                   mumps_data->info[8]);
    Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                   "Number of integers for MUMPS to hold factorization (INFO(10)) = %d\n",
                   mumps_data->info[9]);

    if (error == -10) {//system is singular
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                     "MUMPS returned INFO(1) = %d matrix is singular.\n",error);
      return SYMSOLVER_SINGULAR;
    }

    negevals_ = mumps_data->infog[11];

    if (error == -13) {
      Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
                     "MUMPS returned INFO(1) =%d - out of memory when trying to allocate %d %s.\nIn some cases it helps to decrease the value of the option \"mumps_mem_percent\".\n",
                     error, mumps_data->info[1] < 0 ? -mumps_data->info[1] : mumps_data->info[1], mumps_data->info[1] < 0 ? "MB" : "bytes");
      return SYMSOLVER_FATAL_ERROR;
    }
    if (error < 0) {//some other error
      Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
                     "MUMPS returned INFO(1) =%d MUMPS failure.\n",
                     error);
      return SYMSOLVER_FATAL_ERROR;
    }

    if (check_NegEVals && (numberOfNegEVals!=negevals_)) {
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                     "In CustomSolverInterface::Factorization: negevals_ = %d, but numberOfNegEVals = %d\n",
                     negevals_, numberOfNegEVals);
      return SYMSOLVER_WRONG_INERTIA;
    }

    return SYMSOLVER_SUCCESS;
  }

  ESymSolverStatus CustomSolverInterface::Solve(Index nrhs, double *rhs_vals)
  {
    DBG_START_METH("CustomSolverInterface::Solve", dbg_verbosity);
    DMUMPS_STRUC_C* mumps_data = (DMUMPS_STRUC_C*)mumps_ptr_;
    ESymSolverStatus retval = SYMSOLVER_SUCCESS;
    if (HaveIpData()) {
      IpData().TimingStats().LinearSystemBackSolve().Start();
    }    
    for (Index i = 0; i < nrhs; i++)
    {
      Index offset = i * mumps_data->n;
      mumps_data->rhs = &(rhs_vals[offset]);
      mumps_data->job = 3; //solve
      Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                     "Calling MUMPS-3 for solve at cpu time %10.3f (wall %10.3f).\n", CpuTime(), WallclockTime());
      dmumps_c(mumps_data);
      Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                     "Done with MUMPS-3 for solve at cpu time %10.3f (wall %10.3f).\n", CpuTime(), WallclockTime());
      int error = mumps_data->info[0];
      if (error < 0)
      {
        Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
                       "Error=%d returned from MUMPS in Solve.\n",
                       error);
        retval = SYMSOLVER_FATAL_ERROR;
      }
    }
    if (HaveIpData()) {
      IpData().TimingStats().LinearSystemBackSolve().End();
    }

    // cout<<"$$$$$$$$$$$$$$$$$$$$$$$$$$"<<endl;
    // for (int i=0; i<40; i++) {
    //   printf("%d ", mumps_data->icntl[i]);
    // }
    // printf("\n");
    // for (int i=0; i<5; i++) {
    //   printf("%25.15e\n", mumps_data->cntl[i]);
    // }
    // dump_matrix(mumps_data);
    // cout << "三元组元素：" << endl;
    // for (int i = 0; i < mumps_data->nz; i++)
    // {
    //   // cout << irn_unique[i] << " " << jcn_unique[i] << " " << a_unique[i] << endl;
    //   cout << mumps_data->irn[i] << " " << mumps_data->jcn[i] << " " << mumps_data->a[i] << endl;
    // }
    // cout << "%%%%%%%%%%%%%%%%" << endl;

    // for (int i = 0; i < mumps_data->nz; i++)
    // {
    //   cout << mumps_data->irn[i] << " " << mumps_data->jcn[i] << " " << mumps_data->a[i] << endl;
    //   // cout << mumps_data->irn[i]<<mumps_data->a[i] << endl;
    //   // cout << result[i] << endl;
    // }
    // cout << endl;
    // for (int i = 0; i < mumps_data->n; i++)
    // {
    //   cout << rhs_vals[i] << endl;
    // }
    // // cout << "&&&&&&&&&&&&&&&&" << endl;
    
    // count_test++;
    // if(count_test==2) while(1);
    return retval;
  }

  void CustomSolverInterface::Solve2(int m,
                                     int n,
                                     int nnz,
                                     double *a,
                                     int *asub,
                                     int *xa,
                                     double *b,
                                     int *perm_c,
                                     int *etree)
  {
    char equed[1];
    yes_no_t equil;
    trans_t trans;
    SuperMatrix A, L, U;
    SuperMatrix B, X;
    GlobalLU_t Glu; /* facilitate multiple factorizations with
                                       SamePattern_SameRowPerm            */
    int *perm_r;    /* row permutations from partial pivoting */
    void *work;
    int info, lwork, nrhs;
    int i;
    double *rhsb, *rhsx;
    double *R, *C;
    double *ferr, *berr;
    double u, rpg, rcond;

    mem_usage_t mem_usage;
    superlu_options_t options;
    SuperLUStat_t stat;

    // Defaults
    lwork = 0;
    nrhs = 1;
    equil = YES;
    u = 1.0;
    trans = NOTRANS;
    set_default_options(&options);

    options.Equil = equil;
    options.DiagPivotThresh = u;
    options.Trans = trans;

    /* Add more functionalities that the defaults. */
    options.PivotGrowth = YES;       /* Compute reciprocal pivot growth */
    options.ConditionNumber = YES;   /* Compute reciprocal condition number */
    options.IterRefine = SLU_DOUBLE; /* Perform double-precision refinement */

    if (lwork > 0)
    {
      work = SUPERLU_MALLOC(lwork);
      if (!work)
      {
        ABORT("DLINSOLX: cannot allocate work[]");
      }
    }

    /* Initialize matrix A. */
    dCreate_CompCol_Matrix(&A, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);

    /* Create right-hand side matrix B and X. */
    if (!(rhsb = doubleMalloc(m * nrhs)))
      ABORT("Malloc fails for rhsb[].");
    if (!(rhsx = doubleMalloc(m * nrhs)))
      ABORT("Malloc fails for rhsx[].");
    // if (!(sol = doubleMalloc(m * nrhs)))
    //   ABORT("Malloc fails for sol[].");

    for (i = 0; i < m; ++i)
    {
      rhsb[i] = b[i];
      rhsx[i] = b[i];
    }
    dCreate_Dense_Matrix(&B, m, nrhs, rhsb, m, SLU_DN, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&X, m, nrhs, rhsx, m, SLU_DN, SLU_D, SLU_GE);

    if (!(perm_r = intMalloc(m)))
      ABORT("Malloc fails for perm_r[].");
    if (!(R = (double *)SUPERLU_MALLOC(A.nrow * sizeof(double))))
      ABORT("SUPERLU_MALLOC fails for R[].");
    if (!(C = (double *)SUPERLU_MALLOC(A.ncol * sizeof(double))))
      ABORT("SUPERLU_MALLOC fails for C[].");
    if (!(ferr = (double *)SUPERLU_MALLOC(nrhs * sizeof(double))))
      ABORT("SUPERLU_MALLOC fails for ferr[].");
    if (!(berr = (double *)SUPERLU_MALLOC(nrhs * sizeof(double))))
      ABORT("SUPERLU_MALLOC fails for berr[].");

    /* Initialize the statistics variables. */
    StatInit(&stat);

    /* ------------------------------------------------------------
           WE SOLVE THE LINEAR SYSTEM FOR THE FIRST TIME: AX = B
           ------------------------------------------------------------*/
    dgssvx(&options, &A, perm_c, perm_r, etree, equed, R, C,
           &L, &U, work, lwork, &B, &X, &rpg, &rcond, ferr, berr,
           &Glu, &mem_usage, &stat, &info);

    if (info == 0 || info == n + 1)
    {
      for (int i = 0; i < n; i++)
        b[i] = ((double *)((DNformat *)X.Store)->nzval)[i];
    }
    else if (info > 0 && lwork == -1)
    {
      printf("** Estimated memory: %d bytes\n", info - n);
    }

    //if ( options.PrintStat )
    //	StatPrint(&stat);
    StatFree(&stat);

    SUPERLU_FREE(rhsb);
    SUPERLU_FREE(rhsx);
    SUPERLU_FREE(perm_r);
    SUPERLU_FREE(R);
    SUPERLU_FREE(C);
    SUPERLU_FREE(ferr);
    SUPERLU_FREE(berr);
    Destroy_CompCol_Matrix(&A);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperMatrix_Store(&X);
    if (lwork == 0)
    {
      Destroy_SuperNode_Matrix(&L);
      Destroy_CompCol_Matrix(&U);
    }
    else if (lwork > 0)
    {
      SUPERLU_FREE(work);
    }
  }

  void CustomSolverInterface::Solve3(int m,
                                     int n,
                                     int nnz,
                                     double *a,
                                     int *asub,
                                     int *xa,
                                     double *b,
                                     int *perm_c,
                                     int *etree)
  {
    char equed[1];
    yes_no_t equil;
    trans_t trans;
    SuperMatrix A1, L, U;
    SuperMatrix B, X;
    GlobalLU_t Glu; /* facilitate multiple factorizations with
                                       SamePattern_SameRowPerm            */
    int *perm_r;    /* row permutations from partial pivoting */
    void *work;
    int info, lwork, nrhs;
    int i;
    double *rhsb, *rhsx, *sol;
    double *R, *C;
    double *ferr, *berr;
    double u, rpg, rcond;

    mem_usage_t mem_usage;
    superlu_options_t options;
    SuperLUStat_t stat;

    // Defaults
    lwork = 0;
    nrhs = 1;
    equil = YES;
    u = 1.0;
    trans = NOTRANS;
    set_default_options(&options);

    options.Equil = equil;
    options.DiagPivotThresh = u;
    options.Trans = trans;

    /* Add more functionalities that the defaults. */
    options.PivotGrowth = YES;       /* Compute reciprocal pivot growth */
    options.ConditionNumber = YES;   /* Compute reciprocal condition number */
    options.IterRefine = SLU_DOUBLE; /* Perform double-precision refinement */

    if (lwork > 0)
    {
      work = SUPERLU_MALLOC(lwork);
      if (!work)
      {
        ABORT("DLINSOLX: cannot allocate work[]");
      }
    }

    /* Initialize matrix A1. */
    dCreate_CompCol_Matrix(&A1, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);

    /* Create right-hand side matrix B and X. */
    if (!(rhsb = doubleMalloc(m * nrhs)))
      ABORT("Malloc fails for rhsb[].");
    if (!(rhsx = doubleMalloc(m * nrhs)))
      ABORT("Malloc fails for rhsx[].");
    if (!(sol = doubleMalloc(n * nrhs)))
      ABORT("Malloc fails for rhsx[].");

    for (i = 0; i < m; ++i)
    {
      rhsb[i] = b[i];
      rhsx[i] = b[i];
    }
    dCreate_Dense_Matrix(&B, m, nrhs, rhsb, m, SLU_DN, SLU_D, SLU_GE);
    dCreate_Dense_Matrix(&X, m, nrhs, rhsx, m, SLU_DN, SLU_D, SLU_GE);

    if (!(perm_r = intMalloc(m)))
      ABORT("Malloc fails for perm_r[].");
    if (!(R = (double *)SUPERLU_MALLOC(A1.nrow * sizeof(double))))
      ABORT("SUPERLU_MALLOC fails for R[].");
    if (!(C = (double *)SUPERLU_MALLOC(A1.ncol * sizeof(double))))
      ABORT("SUPERLU_MALLOC fails for C[].");
    if (!(ferr = (double *)SUPERLU_MALLOC(nrhs * sizeof(double))))
      ABORT("SUPERLU_MALLOC fails for ferr[].");
    if (!(berr = (double *)SUPERLU_MALLOC(nrhs * sizeof(double))))
      ABORT("SUPERLU_MALLOC fails for berr[].");

    options.Fact = SamePattern; //稀疏结构相同，复用perm_c

    /* Initialize the statistics variables. */
    StatInit(&stat);

    /* ------------------------------------------------------------
           WE SOLVE THE LINEAR SYSTEM FOR THE FIRST TIME: A1X = B
           ------------------------------------------------------------*/
    dgssvx(&options, &A1, perm_c, perm_r, etree, equed, R, C,
           &L, &U, work, lwork, &B, &X, &rpg, &rcond, ferr, berr,
           &Glu, &mem_usage, &stat, &info);
    if (info == 0 || info == n + 1)
    {
      for (int i = 0; i < n; i++)
        b[i] = ((double *)((DNformat *)X.Store)->nzval)[i];
    }
    else if (info > 0 && lwork == -1)
    {
      printf("** Estimated memory: %d bytes\n", info - n);
    }

    //if ( options.PrintStat )
    //	StatPrint(&stat);
    StatFree(&stat);

    SUPERLU_FREE(rhsb);
    SUPERLU_FREE(rhsx);
    SUPERLU_FREE(perm_r);
    SUPERLU_FREE(R);
    SUPERLU_FREE(C);
    SUPERLU_FREE(ferr);
    SUPERLU_FREE(berr);
    Destroy_CompCol_Matrix(&A1);
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperMatrix_Store(&X);
    if (lwork == 0)
    {
      Destroy_SuperNode_Matrix(&L);
      Destroy_CompCol_Matrix(&U);
    }
    else if (lwork > 0)
    {
      SUPERLU_FREE(work);
    }
  }

  Index CustomSolverInterface::NumberOfNegEVals() const
  {
    DBG_START_METH("CustomSolverInterface::NumberOfNegEVals", dbg_verbosity);
    DBG_ASSERT(negevals_ >= 0);
    return negevals_;
  }

  bool CustomSolverInterface::IncreaseQuality()
  {
    DBG_START_METH("MumpsTSolverInterface::IncreaseQuality",dbg_verbosity);
    if (pivtol_ == pivtolmax_) {
      return false;
    }
    pivtol_changed_ = true;

    Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                   "Increasing pivot tolerance for MUMPS from %7.2e ",
                   pivtol_);

    //this is a more aggresive update then MA27
    //this should be tuned
    pivtol_ = Min(pivtolmax_, pow(pivtol_,0.5));
    Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                   "to %7.2e.\n",
                   pivtol_);
    return true;
  }

  bool CustomSolverInterface::ProvidesDegeneracyDetection() const
  {
    return true;
  }

  ESymSolverStatus CustomSolverInterface::
  DetermineDependentRows(const Index* ia, const Index* ja,
                         std::list<Index>& c_deps)
  {
    DBG_START_METH("CustomSolverInterface::DetermineDependentRows",
                   dbg_verbosity);
    DMUMPS_STRUC_C* mumps_data = (DMUMPS_STRUC_C*)mumps_ptr_;

    c_deps.clear();

    ESymSolverStatus retval;
    // Do the symbolic facotrization if it hasn't been done yet
    if (!have_symbolic_factorization_) {
      const Index mumps_permuting_scaling_orig = mumps_permuting_scaling_;
      const Index mumps_scaling_orig = mumps_scaling_;
      mumps_permuting_scaling_ = 0;
      mumps_scaling_ = 6;
      retval = SymbolicFactorization();
      mumps_permuting_scaling_ = mumps_permuting_scaling_orig;
      mumps_scaling_ = mumps_scaling_orig;
      if (retval != SYMSOLVER_SUCCESS ) {
        return retval;
      }
      have_symbolic_factorization_ = true;
    }
    // perform the factorization, in order to find dependent rows/columns

    //Set flags to ask MUMPS for checking linearly dependent rows
    mumps_data->icntl[23] = 1;
    mumps_data->cntl[2] = mumps_dep_tol_;
    mumps_data->job = 2;//numerical factorization

    dump_matrix(mumps_data);
    dmumps_c(mumps_data);
    int error = mumps_data->info[0];

    //Check for errors
    if (error == -8 || error == -9) {//not enough memory
      const Index trycount_max = 20;
      for (int trycount=0; trycount<trycount_max; trycount++) {
        Jnlst().Printf(J_WARNING, J_LINEAR_ALGEBRA,
                       "MUMPS returned INFO(1) = %d and requires more memory, reallocating.  Attempt %d\n",
                       error,trycount+1);
        Jnlst().Printf(J_WARNING, J_LINEAR_ALGEBRA,
                       "  Increasing icntl[13] from %d to ", mumps_data->icntl[13]);
        double mem_percent = mumps_data->icntl[13];
        mumps_data->icntl[13] = (Index)(2.0 * mem_percent);
        Jnlst().Printf(J_WARNING, J_LINEAR_ALGEBRA, "%d.\n", mumps_data->icntl[13]);

        dump_matrix(mumps_data);
        dmumps_c(mumps_data);
        error = mumps_data->info[0];
        if (error != -8 && error != -9)
          break;
      }
      if (error == -8 || error == -9) {
        Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
                       "MUMPS was not able to obtain enough memory.\n");
        // Reset flags
        mumps_data->icntl[23] = 0;
        return SYMSOLVER_FATAL_ERROR;
      }
    }

    // Reset flags
    mumps_data->icntl[23] = 0;

    if (error < 0) {//some other error
      Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
                     "MUMPS returned INFO(1) =%d MUMPS failure.\n",
                     error);
      return SYMSOLVER_FATAL_ERROR;
    }

    const Index n_deps = mumps_data->infog[27];
    for (Index i=0; i<n_deps; i++) {
      c_deps.push_back(mumps_data->pivnul_list[i]-1);
    }

    return SYMSOLVER_SUCCESS;
  }

}//end Ipopt namespace



