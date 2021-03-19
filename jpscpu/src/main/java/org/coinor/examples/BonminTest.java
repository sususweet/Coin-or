package org.coinor.examples;

import org.coinor.Bonmin;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 2014/6/5
 */
public class BonminTest extends Bonmin {

    /**
     * Main function for running this example.
     */
    public static void main(String[] args) {
        int count = 1;
        BonminTest[] myBonmin = new BonminTest[count];
        for (int i = 0; i < count; i++) {
            // Create the problem
            myBonmin[i] = new BonminTest();
            BonminTest bonminInstance = myBonmin[i];
            //add options, the same with IPOPT.
            //bonminInstance.setStringOption("bonmin.algorithm","B-BB");
            bonminInstance.setStringOption("bonmin.algorithm","B-OA");
            //bonminInstance.setStringOption("bonmin.algorithm","B-QG");
            //bonminInstance.setStringOption("bonmin.algorithm","B-Hyb");
            //bonminInstance.setStringOption("bonmin.algorithm","B-Ecp");
            //bonminInstance.setStringOption("bonmin.algorithm","B-iFP");

            bonminInstance.OptimizeMINLP();
            //Below see results
            double x[] = bonminInstance.getState();
            HS071.print(x, "Optimal Solution:");

            double[] MLB = bonminInstance.getMultLowerBounds();
            HS071.print(MLB, "Multipler LowerBounds:");

            double[] MUB = bonminInstance.getMultUpperBounds();
            HS071.print(MUB, "Multipler UpperBounds:");

            double obj = bonminInstance.getObjVal();
            System.out.println("Obj Value=" + obj);

            double[] constraints = bonminInstance.getMultConstraints();
            HS071.print(constraints, "G(x):");
            double[] lam = bonminInstance.getMultConstraints();
            HS071.print(lam, "Constraints Multipler");
            bonminInstance.dispose();
        }
        System.out.println("end..");
    }

    public BonminTest() {
        createBonmin(4, 3, 7, 2, FORTRAN_STYLE);
    }

    @Override
    protected boolean get_variables_types(int n, int[] var_types) {
        var_types[0] = BONMIN_BINARY;
        var_types[1] = BONMIN_CONTINUOUS;
        var_types[2] = BONMIN_CONTINUOUS;
        var_types[3] = BONMIN_INTEGER;
        return true;
    }

    @Override
    protected boolean get_variables_linearity(int n, int[] var_types) {
        var_types[0] = IPOPT_TNLP_LINEAR;
        var_types[1] = IPOPT_TNLP_NON_LINEAR;
        var_types[2] = IPOPT_TNLP_NON_LINEAR;
        var_types[3] = IPOPT_TNLP_LINEAR;
        return true;
    }

    @Override
    protected boolean get_constraints_linearity(int m, int[] const_types) {
        const_types[0] = IPOPT_TNLP_NON_LINEAR;
        const_types[1] = IPOPT_TNLP_LINEAR;
        const_types[2] = IPOPT_TNLP_LINEAR;
        return true;
    }

    @Override
    protected boolean get_bounds_info(int n, double[] x_l, double[] x_u, int m, double[] g_l, double[] g_u) {
        x_l[0] = 0.;
        x_u[0] = 1.;

        x_l[1] = 0.;
        x_u[1] = 1e5;

        x_l[2] = 0.;
        x_u[2] = 1e5;

        x_l[3] = 0;
        x_u[3] = 5;

        g_l[0] = -1e5;
        g_u[0] = 1. / 4.;

        g_l[1] = -1e5;
        g_u[1] = 0;

        g_l[2] = -1e5;
        g_u[2] = 2;
        return true;
    }

    @Override
    protected boolean get_starting_point(int n, boolean init_x, double[] x, boolean init_z, double[] z_L, double[] z_U, int m, boolean init_lambda, double[] lambda) {
        x[0] = 0;
        x[1] = 0;
        x[2] = 0;
        x[3] = 0;
        return true;
    }

    @Override
    protected boolean eval_f(int n, double[] x, boolean new_x, double[] obj_value) {
        obj_value[0] = -x[0] - x[1] - x[2];
        return true;
    }

    @Override
    protected boolean eval_grad_f(int n, double[] x, boolean new_x, double[] grad_f) {
        grad_f[0] = -1.;
        grad_f[1] = -1.;
        grad_f[2] = -1.;
        grad_f[3] = 0.;
        return true;
    }

    @Override
    protected boolean eval_g(int n, double[] x, boolean new_x, int m, double[] g) {
        g[0] = (x[1] - 1. / 2.) * (x[1] - 1. / 2.) + (x[2] - 1. / 2.) * (x[2] - 1. / 2.);
        g[1] = x[0] - x[1];
        g[2] = x[0] + x[2] + x[3];

        return true;
    }

    @Override
    protected boolean eval_jac_g(int n, double[] x, boolean new_x, int m, int nele_jac, int[] iRow, int[] jCol, double[] values) {
        if (values == null) {
            iRow[0] = 2;
            jCol[0] = 1;

            iRow[1] = 3;
            jCol[1] = 1;

            iRow[2] = 1;
            jCol[2] = 2;

            iRow[3] = 2;
            jCol[3] = 2;

            iRow[4] = 1;
            jCol[4] = 3;

            iRow[5] = 3;
            jCol[5] = 3;

            iRow[6] = 3;
            jCol[6] = 4;
            return true;
        } else {
            values[0] = 1.;
            values[1] = 1;

            values[2] = 2 * x[1] - 1;
            values[3] = -1.;

            values[4] = 2 * x[2] - 1;
            values[5] = 1.;

            values[6] = 1.;

            return true;
        }
    }

    @Override
    protected boolean eval_h(int n, double[] x, boolean new_x, double obj_factor, int m, double[] lambda, boolean new_lambda, int nele_hess, int[] iRow, int[] jCol, double[] values) {
        if (values == null) {
            iRow[0] = 2;
            jCol[0] = 2;

            iRow[1] = 3;
            jCol[1] = 3;
        } else {
            values[0] = 2 * lambda[0];
            values[1] = 2 * lambda[0];
        }
        return true;
    }
}
