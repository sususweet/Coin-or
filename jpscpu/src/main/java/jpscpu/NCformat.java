package jpscpu;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 13-9-21
 */
public class NCformat {
    int nnz;	    /* number of nonzeros in the matrix */
    double[] nzval;    /* pointer to array of nonzero values, packed by column */
    int[] rowind; /* pointer to array of row indices of the nonzeros */
    int[] colptr; /* pointer to array of beginning of columns in nzval[]
                 and rowind[]  */
                      /* Note:
  		       Zero-based indexing is used;
  		       colptr[] has ncol+1 entries, the last one pointing
  		       beyond the last column, so that colptr[ncol] = nnz. */

    public int getNnz() {
        return nnz;
    }

    public void setNnz(int nnz) {
        this.nnz = nnz;
    }

    public double[] getNzval() {
        return nzval;
    }

    public void setNzval(double[] nzval) {
        this.nzval = nzval;
    }

    public int[] getRowind() {
        return rowind;
    }

    public void setRowind(int[] rowind) {
        this.rowind = rowind;
    }

    public int[] getColptr() {
        return colptr;
    }

    public void setColptr(int[] colptr) {
        this.colptr = colptr;
    }
}
