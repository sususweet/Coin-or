package org.coinor;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 2014/6/4
 */
public abstract class Bonmin extends Ipopt {
    public static final int BONMIN_CONTINUOUS = 0;
    public static final int BONMIN_BINARY = 1;
    public static final int BONMIN_INTEGER = 2;
    public static final int IPOPT_TNLP_LINEAR = 0;
    public static final int IPOPT_TNLP_NON_LINEAR = 1;

    /**
     * Native function should not be used directly
     */
    private native boolean AddBonminIntOption(long Bonmin, String keyword, int val);

    /**
     * Native function should not be used directly
     */
    private native boolean AddBonminNumOption(long Bonmin, String keyword, double val);

    /**
     * Native function should not be used directly
     */
    private native boolean AddBonminStrOption(long bonmin, String keyword, String val);

    /**
     * Native function should not be used directly
     */
    private native long CreateBonminProblem(int n, int m, int nele_jac, int nele_hess, int index_style);

    /* Native function should not be used directly */
    private native void FreeBonminProblem(long bonmin);

    /**
     * Native function should not be used directly
     */
    private native int OptimizeTMINLP(long bonmin, String outfilename,
                                      double x[], double g[],
                                      double obj_val[], double mult_g[], double mult_x_L[], double mult_x_U[],
                                      double callback_grad_f[], double callback_jac_g[], double callback_hess[]);
    //@}


    /**
     * Pass the type of the variables (INTEGER, BINARY, CONTINUOUS) to the optimizer.
     * \param n size of var_types (has to be equal to the number of variables in the problem)
     * \param var_types types of the variables (has to be filled by function).
     */
    abstract protected boolean get_variables_types(int n, int[] var_types);

    /**
     * Pass info about linear and nonlinear variables.
     */
    abstract protected boolean get_variables_linearity(int n, int[] var_types);

    /**
     * Pass the type of the constraints (LINEAR, NON_LINEAR) to the optimizer.
     * \param m size of const_types (has to be equal to the number of constraints in the problem)
     * \param const_types types of the constraints (has to be filled by function).
     */
    abstract protected boolean get_constraints_linearity(int m, int[] const_types);

    private long bonmin;

    public void dispose() {
        // dispose the native implementation
        if (bonmin != 0) {
            FreeBonminProblem(bonmin);
            bonmin = 0;
        } else
            super.dispose();
    }

    /**
     * Create a new problem. the use is the same as get_nlp_info, change the name for clarity in java.
     *
     * @param n           the number of variables in the problem.
     * @param m           the number of constraints in the problem.
     * @param nele_jac    the number of nonzero entries in the Jacobian.
     * @param nele_hess   the number of nonzero entries in the essian.
     * @param index_style the numbering style used for row/col entries in the sparse matrix format(0 for
     *                    C_STYLE, 1 for FORTRAN_STYLE).
     * @return true means success, false means fail!
     */
    public boolean createBonmin(int n, int m, int nele_jac, int nele_hess, int index_style) {
        // delete any previously created native memory
        dispose();

        x = new double[n];
        // allocate the callback arguments
        callback_grad_f = new double[n];
        callback_jac_g = new double[nele_jac];
        callback_hess = new double[nele_hess];

        // the multiplier
        mult_x_U = new double[n];
        mult_x_L = new double[n];
        g = new double[m];
        mult_g = new double[m];

        //	Create the optimization problem and return a pointer to it
        bonmin = CreateBonminProblem(n, m, nele_jac, nele_hess, index_style);

        //System.out.println("Finish Java Obj");
        return bonmin != 0;
    }

    /**
     * Function for adding an integer option.
     * <p/>
     * The valid keywords are public static members of this class, with names
     * beginning with <code>KEY_</code>, e.g, {@link #KEY_TOL}.
     * For more details about the valid options check the Ipopt documentation.
     *
     * @param keyword the option keyword
     * @param val     the value
     * @return false if the option could not be set (e.g., if keyword is unknown)
     */
    protected boolean setIntegerOption(String keyword, int val) {
        if (bonmin != 0)
            return AddBonminIntOption(bonmin, keyword, val);
        else
            return super.setIntegerOption(keyword, val);
    }

    /**
     * Function for adding a number option.
     *
     * @param keyword the option keyword
     * @param val     the value
     * @return false if the option could not be set (e.g., if keyword is unknown)
     * @see #setIntegerOption(String, int)
     */
    protected boolean setNumericOption(String keyword, double val) {
        if (bonmin != 0)
            return AddBonminNumOption(bonmin, keyword, val);
        else
            return super.setNumericOption(keyword, val);
    }

    /**
     * Function for adding a string option.
     *
     * @param keyword the option keyword
     * @param val     the value
     * @return false if the option could not be set (e.g., if keyword is unknown)
     * @see #setIntegerOption(String, int)
     */
    public boolean setStringOption(String keyword, String val) {
        if (bonmin != 0)
            return AddBonminStrOption(bonmin, keyword, val.toLowerCase());
        else
            return super.setStringOption(keyword, val);
    }

    /**
     * This function actually solve the problem.
     * <p/>
     * The solve status returned is one of the constant fields of this class,
     * e.g. SOLVE_SUCCEEDED. For more details about the valid solve status
     * check the Ipopt documentation or the <code>ReturnCodes_inc.h<\code>
     * which is installed in the Ipopt include directory.	 *
     *
     * @return the solve status
     * @see #getStatus()
     */
    public int OptimizeMINLP() {
        String outfilename = "";
        this.status = this.OptimizeTMINLP(bonmin, outfilename,
                x, g, obj_val, mult_g, mult_x_L, mult_x_U,
                callback_grad_f, callback_jac_g, callback_hess);
        return this.status;
    }
}
