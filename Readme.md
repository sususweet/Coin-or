Coin-or 规划问题求解库
===========

https://projects.coin-or.org/BuildTools

Windows 下 MingGW 编译流程如下：

+ 下载上面的源代码，并按照 `ThirdParty` 目录里的提示下载第三方库。

+ CYGWIN 定位到代码存放目录，按顺序执行下面的命令即可

```
configure --enable-shared --enable-static --enable-dependency-linking lt_cv_deplibs_check_method=pass_all
make
```

对于 32 位的 MingGW 编译器，在 `configure` 后增加选项

```
--host=i686-pc-mingw32
```

对于 64 位的 MingGW 编译器，在 `configure` 后增加选项

```
--host=x86_64-w64-mingw32
```

+ 最后即可得到所有的动态链接库和静态链接库。