package jpscpu;

import java.io.*;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 13-7-10
 */
public class SoFileLoader {

    public static final String systemType = System.getProperty("os.name");
    public static boolean isLoaded = false;

    public static void loadSoFiles() {
        if (isLoaded)
            return;
        try {
            isLoaded = true;
            if (systemType.toLowerCase().contains("win")) {
                //系统库
                writeSoFile("/win32/libstdc++-6.dll", "libstdc++-6.dll");
                writeSoFile("/win32/libgfortran-3.dll", "libgfortran-3.dll");
                writeSoFile("/win32/libgcc_s_dw2-1.dll", "libgcc_s_dw2-1.dll");
                writeSoFile("/win32/libquadmath-0.dll", "libquadmath-0.dll");

                //COIN常用库
                //writeSoFile("/win32/libcoinblas.a", "libcoinblas.a");
                //writeSoFile("/win32/libcoinlapack.a", "libcoinlapack.a");
                //writeSoFile("/win32/libcoinhsl.a", "libcoinhsl.a");
                writeSoFile("/win32/libipopt.dll", "libipopt.dll");
                writeSoFile("/win32/libCoinUtils.dll", "libCoinUtils.dll");
                writeSoFile("/win32/libClp.dll", "libClp.dll");
                writeSoFile("/win32/libOsi.dll", "libOsi.dll");
                writeSoFile("/win32/libCgl.dll", "libCgl.dll");
                writeSoFile("/win32/libCbc.dll", "libCbc.dll");
                writeSoFile("/win32/libSym.dll", "libSym.dll");
                writeSoFile("/win32/libBonmin.dll", "libBonmin.dll");

                //SuperLU库
                //writeSoFile("/win32/libsuperlu_4.3.a", "libsuperlu_4.3.a");

                //加载
                loadLib("/win32/libjpscpu.dll", "libjpscpu.dll");
            } else if (systemType.toLowerCase().contains("linux")) {
                //在Linux下要求所有.so文件都放在LD_LIBRARY_PATH能够找到的地方
                System.loadLibrary("jpscpu");
            }
        } catch (Exception e) {
            isLoaded = false;
            System.err.println("load jni error!");
        }
    }

    private static void loadLib(String libFullName, String simpleName) throws IOException {
        File f = writeSoFile(libFullName, simpleName);
        System.load(f.toString());
    }

    private static File writeSoFile(String libFullName, String simpleName) throws IOException {
        String nativeTempDir = System.getProperty("user.dir");

        InputStream in = null;
        BufferedInputStream reader;
        FileOutputStream writer = null;

        File extractedLibFile = new File(nativeTempDir + File.separator + simpleName);
        if (!extractedLibFile.exists()) {
            try {
                in = SoFileLoader.class.getResourceAsStream(libFullName);
                if (in == null)
                    in = SoFileLoader.class.getResourceAsStream(libFullName);
                SoFileLoader.class.getResource(libFullName);
                reader = new BufferedInputStream(in);
                writer = new FileOutputStream(extractedLibFile);

                byte[] buffer = new byte[1024];

                while (reader.read(buffer) > 0) {
                    writer.write(buffer);
                    buffer = new byte[1024];
                }
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                if (in != null)
                    in.close();
                if (writer != null)
                    writer.close();
            }
        }
        extractedLibFile.deleteOnExit();
        return extractedLibFile;
    }
}
