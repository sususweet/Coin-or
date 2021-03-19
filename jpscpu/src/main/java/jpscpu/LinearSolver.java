package jpscpu;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import zju.matrix.ASparseMatrixLink2D;
import zju.util.ColtMatrixUtil;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 13-8-19
 */
public class LinearSolver {

    public static final int SUPERLU_DRIVE_0 = 0;
    public static final int SUPERLU_DRIVE_1 = 1;
    public static final int MLP_DRIVE_CBC = 2;
    public static final int MLP_DRIVE_SYM = 3;

    private int drive = SUPERLU_DRIVE_0;

    private int[] perm_c, perm_r, etree;

    private int m, n, nnz;

    private double[] a;

    private int[] asub, xa;

    private SCformat L;

    private NCformat U;

    private double[] R, C;

    private int[] equed_int;

    static {
        SoFileLoader.loadSoFiles();
    }

    /**
     * 计算  Ax = b,使用dgssv引擎
     *
     * @param m    A的行数
     * @param n    A的列数
     * @param nnz  A的非零元个数
     * @param a    A的非零元的数值
     * @param asub A的非零元行号
     * @param xa   xa的第i个元素表示前i行共有多少个非零元
     * @param b    向量b
     * @return 计算结果x
     */
    private native double[] solve0(int m, int n, int nnz, double[] a, int[] asub, int[] xa, double[] b);

    /**
     * 计算  Ax = b,使用dgssvx引擎
     *
     * @param m    A的行数
     * @param n    A的列数
     * @param nnz  A的非零元个数
     * @param a    A的非零元的数值
     * @param asub A的非零元行号
     * @param xa   xa的第i个元素表示前i行共有多少个非零元
     * @param b    向量b
     * @return 计算结果x
     */
    private native double[] solve1(int m, int n, int nnz, double[] a, int[] asub, int[] xa, double[] b);

    /**
     * <br>计算  Ax = b,使用dgssvx引擎, solver2和solve3配合适合多次计算，而A的结构不变，只是值变化的情况</br>
     * <br>solve2是第一次调用时使用，perm_c和etree用于存储第一次对A进行LU分解的结构信息，在cpp中进行赋值</br>
     *
     * @param m      A的行数
     * @param n      A的列数
     * @param nnz    A的非零元个数
     * @param a      A的非零元的数值
     * @param asub   A的非零元行号
     * @param xa     xa的第i个元素表示前i行共有多少个非零元
     * @param b      向量b
     * @param perm_c 用于存储A分解后的信息
     * @param etree  用于存储A分解后的信息
     * @return 计算结果x
     */
    private native double[] solve2(int m, int n, int nnz, double[] a, int[] asub, int[] xa, double[] b, int[] perm_c, int[] etree);

    /**
     * <br>计算  Ax = b,使用dgssvx引擎, solver2和solve3配合适合多次计算，而A的结构不变，只是值变化的情况</br>
     * <br>solve3是非第一次调用时使用，perm_c和etree已经存储了第一次对A进行LU分解的结构信息</br>
     *
     * @param m      A的行数
     * @param n      A的列数
     * @param nnz    A的非零元个数
     * @param a      A的非零元的数值
     * @param asub   A的非零元行号
     * @param xa     xa的第i个元素表示前i行共有多少个非零元
     * @param b      向量b
     * @param perm_c 用于存储A分解后的信息
     * @param etree  用于存储A分解后的信息
     * @return 计算结果x
     */
    private native double[] solve3(int m, int n, int nnz, double[] a, int[] asub, int[] xa, double[] b, int[] perm_c, int[] etree);

    /**
     * <br>计算  Ax = b,使用dgssvx引擎, solver4和solve5配合适合多次计算，适合A不变，只是b变化的情况</br>
     * <br>solve4是第一次调用时使用，将LU分解的结果存储在perm_c, perm_r, etree, R, C, L, U中</br>
     *
     * @param m      A的行数
     * @param n      A的列数
     * @param nnz    A的非零元个数
     * @param a      A的非零元的数值
     * @param asub   A的非零元行号
     * @param xa     xa的第i个元素表示前i行共有多少个非零元
     * @param b      向量b
     * @param perm_c 用于存储A分解后的信息
     * @param etree  用于存储A分解后的信息
     * @return 计算结果x
     */
    private native double[] solve4(int m, int n, int nnz, double[] a, int[] asub, int[] xa,
                                   double[] b, int[] perm_c, int[] perm_r, int[] etree,
                                   double[] R, double[] C, SCformat L, NCformat U, int[] equad_int);

    /**
     * <br>计算  Ax = b,使用dgssvx引擎, solver4和solve5配合适合多次计算，适合A不变，只是b变化的情况</br>
     * <br>solve5是非第一次调用时使用，重用perm_c, perm_r, R, C, L, U</br>
     *
     * @param m      A的行数
     * @param n      A的列数
     * @param nnz    A的非零元个数
     * @param a      A的非零元的数值
     * @param asub   A的非零元行号
     * @param xa     xa的第i个元素表示前i行共有多少个非零元
     * @param b      向量b
     * @param perm_c 用于存储A分解后的信息
     * @param etree  用于存储A分解后的信息
     * @return 计算结果x
     */
    private native double[] solve5(int m, int n, int nnz, double[] a, int[] asub, int[] xa, double[] b,
                                   int[] perm_c, int[] perm_r, int[] etree, double[] R, double[] C,
                                   int nnz_L, int nsuper_L, double[] nzval_L, int[] nzval_colptr_L, int[] rowind_L, int[] rowind_colptr_L, int[] col_to_sup_L, int[] sup_to_col_L,
                                   int nnz_U, double[] nzval_U, int[] rowind_U, int[] colptr_U, int[] equad_int);

    /**
     * @param n           变量个数
     * @param m           约束个数
     * @param objValue    目标函数的参数
     * @param columnLower x下限
     * @param columnUpper x上限
     * @param rowLower    约束下限
     * @param rowUpper    约束上限
     * @param element     约束的参数
     * @param column      element中每个参数对应的列数
     * @param starts      每行第一个非零元数在element中的位置
     * @param whichInt    变量中整数的位置
     * @return status
     */
    private native int solveMlpCbc(int n, int m, double objValue[], double columnLower[], double columnUpper[],
                                   double rowLower[], double rowUpper[], double element[],
                                   int column[], int starts[], int whichInt[], double[] result);

    private native int solveMlpSym(int n, int m, double objValue[], double columnLower[], double columnUpper[],
                                   double rowLower[], double rowUpper[], double element[],
                                   int column[], int starts[], int whichInt[], double[] result);

    /**
     * 求解Ax = b, 求解结果存储在right中，第一次求解可以调用该方法
     *
     * @param jacStruc jacobian矩阵的结构
     * @param left     A矩阵
     * @param right    b向量，求解结果存储在该向量中
     * @return right
     */
    public double[] solve(ASparseMatrixLink2D jacStruc, DoubleMatrix2D left, double[] right) {
        m = jacStruc.getM();
        n = jacStruc.getN();
        nnz = jacStruc.getVA().size();
        a = new double[jacStruc.getVA().size()];
        asub = new int[jacStruc.getVA().size()];
        xa = new int[jacStruc.getN() + 1];
        jacStruc.getSluStrucNC(asub, xa);
        ColtMatrixUtil.getSluMatrixNC(left, a, asub, xa);
        return solve(m, n, nnz, a, asub, xa, right);
    }

    /**
     * 求解Ax = b, 求解结果存储在right中，非第一次求解可以调用该方法
     *
     * @param left  A矩阵
     * @param right b向量，求解结果存储在该向量中
     * @return right
     */
    public double[] solve(DoubleMatrix2D left, double[] right) {
        ColtMatrixUtil.getSluMatrixNC(left, a, asub, xa);
        return solve(m, n, nnz, a, asub, xa, right);
    }

    /**
     * 对于Ax=b, A的结构不变且需要多次求解的问题，第一次求解可以使用此方法
     *
     * @param jacStruc A矩阵结构
     * @param left     矩阵A
     * @param b        向量b
     * @return x
     */
    public double[] solve2(ASparseMatrixLink2D jacStruc, DoubleMatrix2D left, double[] b) {
        m = jacStruc.getM();
        n = jacStruc.getN();
        nnz = jacStruc.getVA().size();
        a = new double[jacStruc.getVA().size()];
        asub = new int[jacStruc.getVA().size()];
        xa = new int[jacStruc.getN() + 1];
        jacStruc.getSluStrucNC(asub, xa);
        ColtMatrixUtil.getSluMatrixNC(left, a, asub, xa);
        perm_c = new int[n];
        etree = new int[n];
        return solve2(m, n, nnz, a, asub, xa, b, perm_c, etree);
    }


    /**
     * 对于Ax=b, A的结构不变且需要多次求解的问题
     *
     * @param left        矩阵A
     * @param b           向量b
     * @param isFirstTime 是否是第一次计算
     * @return x
     */
    public double[] solve2(ASparseMatrixLink2D left, double[] b, boolean isFirstTime) {
        if (isFirstTime) {
            m = left.getM();
            n = left.getN();
            nnz = left.getVA().size();
            a = new double[left.getVA().size()];
            asub = new int[left.getVA().size()];
            xa = new int[left.getN() + 1];
            left.getSluStrucNC(a, asub, xa);
            perm_c = new int[n];
            etree = new int[n];
            return solve2(m, n, nnz, a, asub, xa, b, perm_c, etree);
        } else {
            left.getSluStrucNC(a);
            return solve3(m, n, nnz, a, asub, xa, b, perm_c, etree);
        }
    }

    public double[] solve2(DoubleMatrix1D right) {
        return solve2(right.toArray());
    }

    /**
     * 对于Ax=b, A的结构和值均不变且需要多次求解的问题， 非第一次求解可以使用方法
     *
     * @param right 向量b
     * @return x
     */
    public double[] solve2(double[] right) {
        return solve3(m, n, nnz, a, asub, xa, right, perm_c, etree);
    }

    /**
     * 对于Ax=b, A的结构不变,只有值发生变化，且需要多次求解的问题，非第一次求解可以使用方法
     *
     * @param left 矩阵A
     * @param b    向量b
     * @return x
     */
    public double[] solve2(DoubleMatrix2D left, double[] b) {
        ColtMatrixUtil.getSluMatrixNC(left, a, asub, xa);
        return solve3(m, n, nnz, a, asub, xa, b, perm_c, etree);
    }

    public double[] solve(int m, int n, int nnz, double[] a, int[] asub, int[] xa, double[] b) {
        switch (drive) {
            case SUPERLU_DRIVE_0:
                return solve0(m, n, nnz, a, asub, xa, b);
            case SUPERLU_DRIVE_1:
                return solve1(m, n, nnz, a, asub, xa, b);
            default:
                return null;
        }
    }

    /**
     * 对于Ax=b, A不变且需要多次求解的问题，第一次求解可以使用方法
     *
     * @param jacStruc A矩阵结构
     * @param left     矩阵A
     * @param b        向量b
     * @return x
     */
    public double[] solve3(ASparseMatrixLink2D jacStruc, DoubleMatrix2D left, double[] b) {
        m = jacStruc.getM();
        n = jacStruc.getN();
        nnz = jacStruc.getVA().size();
        a = new double[jacStruc.getVA().size()];
        asub = new int[jacStruc.getVA().size()];
        xa = new int[jacStruc.getN() + 1];
        jacStruc.getSluStrucNC(asub, xa);
        ColtMatrixUtil.getSluMatrixNC(left, a, asub, xa);
        L = new SCformat();
        U = new NCformat();
        R = new double[m];
        C = new double[n];
        perm_c = new int[n];
        perm_r = new int[m];
        etree = new int[n];
        equed_int = new int[1];
        return solve4(m, n, nnz, a, asub, xa, b, perm_c, perm_r, etree, R, C, L, U, equed_int);
    }


    /**
     * 对于Ax=b, A不变且需要多次求解的问题
     *
     * @param left 矩阵A
     * @param b    向量b
     * @return x
     */

    public double[] solve3(ASparseMatrixLink2D left, double[] b) {
        m = left.getM();
        n = left.getN();
        nnz = left.getVA().size();
        a = new double[left.getVA().size()];
        asub = new int[left.getVA().size()];
        xa = new int[left.getN() + 1];
        left.getSluStrucNC(a, asub, xa);
        L = new SCformat();
        U = new NCformat();
        R = new double[m];
        C = new double[n];
        perm_c = new int[n];
        perm_r = new int[m];
        etree = new int[n];
        equed_int = new int[1];
        return solve4(m, n, nnz, a, asub, xa, b, perm_c, perm_r, etree, R, C, L, U, equed_int);
    }

    public double[] solve3(DoubleMatrix1D right) {
        return solve3(right.toArray());
    }

    /**
     * 对于Ax=b, A的结构和值均不变且需要多次求解的问题， 非第一次求解可以使用方法
     *
     * @param b 向量b
     * @return x
     */
    public double[] solve3(double[] b) {
        return solve5(m, n, nnz, a, asub, xa, b, perm_c, perm_r, etree, R, C,
                L.getNnz(), L.getNsuper(), L.getNzval(), L.getNzval_colptr(), L.getRowind(),
                L.getRowind_colptr(), L.getCol_to_sup(), L.getSup_to_col(),
                U.getNnz(), U.getNzval(), U.getRowind(), U.getColptr(), equed_int);
    }

    /**
     * @param n           变量个数
     * @param m           约束个数
     * @param objValue    目标函数的参数
     * @param columnLower x下限
     * @param columnUpper x上限
     * @param rowLower    约束下限
     * @param rowUpper    约束上限
     * @param element     约束的参数
     * @param column      element中每个参数对应的列数
     * @param starts      每行第一个非零元数在element中的位置
     * @param whichInt    变量中整数的位置
     * @return status
     */
    public int solveMlp(int n, int m, double objValue[], double columnLower[], double columnUpper[],
                        double rowLower[], double rowUpper[], double element[],
                        int column[], int starts[], int whichInt[], double[] result) {
        if (MLP_DRIVE_SYM == drive) {
            return solveMlpSym(n, m, objValue, columnLower, columnUpper,
                    rowLower, rowUpper, element, column, starts, whichInt, result);
        } else {
            return solveMlpCbc(n, m, objValue, columnLower, columnUpper,
                    rowLower, rowUpper, element, column, starts, whichInt, result);
        }
    }

    public int getDrive() {
        return drive;
    }

    public void setDrive(int drive) {
        this.drive = drive;
    }


}
