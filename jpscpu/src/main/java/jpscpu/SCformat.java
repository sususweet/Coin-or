package jpscpu;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 13-9-21
 */
public class SCformat {
    int nnz;	     /* number of nonzeros in the matrix */
    int nsuper;     /* number of supernodes, minus 1 */
    double[] nzval;       /* pointer to array of nonzero values, packed by column */
    int[] nzval_colptr;/* pointer to array of beginning of columns in nzval[] */
    int[] rowind;     /* pointer to array of compressed row indices of
               rectangular supernodes */
    int[] rowind_colptr;/* pointer to array of beginning of columns in rowind[] */
    int[] col_to_sup;   /* col_to_sup[j] is the supernode number to which column
   			j belongs; mapping from column to supernode number. */
    int[] sup_to_col;   /* sup_to_col[s] points to the start of the s-th
   			supernode; mapping from supernode number to column.
   		        e.g.: col_to_sup: 0 1 2 2 3 3 3 4 4 4 4 4 4 (ncol=12)
   		              sup_to_col: 0 1 2 4 7 12           (nsuper=4) */
                        /* Note:
   		        Zero-based indexing is used;
   		        nzval_colptr[], rowind_colptr[], col_to_sup and
   		        sup_to_col[] have ncol+1 entries, the last one
   		        pointing beyond the last column.
   		        For col_to_sup[], only the first ncol entries are
   		        defined. For sup_to_col[], only the first nsuper+2
   		        entries are defined. */

    public int getNnz() {
        return nnz;
    }

    public void setNnz(int nnz) {
        this.nnz = nnz;
    }

    public int getNsuper() {
        return nsuper;
    }

    public void setNsuper(int nsuper) {
        this.nsuper = nsuper;
    }

    public double[] getNzval() {
        return nzval;
    }

    public void setNzval(double[] nzval) {
        this.nzval = nzval;
    }

    public int[] getNzval_colptr() {
        return nzval_colptr;
    }

    public void setNzval_colptr(int[] nzval_colptr) {
        this.nzval_colptr = nzval_colptr;
    }

    public int[] getRowind() {
        return rowind;
    }

    public void setRowind(int[] rowind) {
        this.rowind = rowind;
    }

    public int[] getRowind_colptr() {
        return rowind_colptr;
    }

    public void setRowind_colptr(int[] rowind_colptr) {
        this.rowind_colptr = rowind_colptr;
    }

    public int[] getCol_to_sup() {
        return col_to_sup;
    }

    public void setCol_to_sup(int[] col_to_sup) {
        this.col_to_sup = col_to_sup;
    }

    public int[] getSup_to_col() {
        return sup_to_col;
    }

    public void setSup_to_col(int[] sup_to_col) {
        this.sup_to_col = sup_to_col;
    }
}
