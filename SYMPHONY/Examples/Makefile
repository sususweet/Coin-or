#===========================================================================#
#                                                                           #
# This file is part of the SYMPHONY MILP Solver Framework.                  #
#                                                                           #
# SYMPHONY was jointly developed by Ted Ralphs (ted@lehigh.edu) and         #
# Laci Ladanyi (ladanyi@us.ibm.com).                                        #
#                                                                           #
# The author of this file is Menal Guzelsoy                                 #
# Modified by tkr July 6, 2010                                              #
#                                                                           #
# (c) Copyright 2006-2015 Lehigh University. All Rights Reserved.           #
#                                                                           #
# This software is licensed under the Eclipse Public License. Please see    #
# accompanying file for terms.                                              #
#                                                                           #
#===========================================================================#

# $Id: Makefile.in 726 2006-04-17 04:16:00Z andreasw $

# The following is necessary under cygwin, if native compilers are used
CYGPATH_W = echo

EXAMPLES = milp milp2 bicriteria sensitivity \
	warm_start1 warm_start2 warm_start3

##############################################################################
# If you wish to use SYMPHONY through the SYMPHONY OSI interface, 
# set USE_OSI_INTERFACE to TRUE below. This will have the examples 
# to call the OSI SYMPHONY interface routines. 
##############################################################################

USE_OSI_INTERFACE = FALSE

ifeq ($(USE_OSI_INTERFACE),TRUE)
EXAMPLES += milpOsi milpOsi2
endif

# Directory with Application files
SYMEXDIR = /cygdrive/h/Java/Bonmin-1.8.6/SYMPHONY/src/../Examples

# Additional compiler flags
ADDINCFLAGS =

##########################################################################
#  Usually, you don't have to change anything below.  Note that if you   #
#  change certain compiler options, you might have to recompile the      #
#  COIN package.                                                         #
##########################################################################
USE_CGL_CUTS = TRUE
CLP_LP_SOLVER = TRUE
GLPK_LP_SOLVER = #TRUE
CPLEX_LP_SOLVER = #TRUE
OSL_LP_SOLVER = #TRUE
SOPLEX_LP_SOLVER = #TRUE
XPRESS_LP_SOLVER = #TRUE

COIN_HAS_PKGCONFIG = TRUE
COIN_CXX_IS_CL = #TRUE

VPATH = $(SYMEXDIR)

WHATTOMAKE = $(addsuffix .exe, $(EXAMPLES))

# C++ Compiler command
CXX = i686-w64-mingw32-g++

# C++ Compiler options
CXXFLAGS = -O3 -pipe -DNDEBUG -pedantic-errors -Wparentheses -Wreturn-type -Wcast-qual -Wall -Wpointer-arith -Wwrite-strings -Wconversion -Wno-unknown-pragmas -Wno-long-long   -DSYMPHONY_BUILD -Wno-unknown-pragmas -Wno-write-strings -Wno-sign-compare -Wno-cast-qual

# additional C++ Compiler options for linking
CXXLINKFLAGS = 

# if USE_OSI_INTERFACE is TRUE, we use osi-sym to get compiler/linker flags from pkg-config, otherwise we use symphony
ifeq ($(USE_OSI_INTERFACE),TRUE)
  ifeq ($(COIN_HAS_PKGCONFIG), TRUE)
    INCL = `PKG_CONFIG_PATH=/cygdrive/h/Java/Bonmin-1.8.6/SYMPHONY/lib64/pkgconfig:/cygdrive/h/Java/Bonmin-1.8.6/SYMPHONY/lib/pkgconfig:/cygdrive/h/Java/Bonmin-1.8.6/SYMPHONY/share/pkgconfig: i686-w64-mingw32-pkg-config --cflags osi-sym`
  else
    INCL = 
  endif

  ifeq ($(COIN_HAS_PKGCONFIG), TRUE)
    LIBS = `PKG_CONFIG_PATH=/cygdrive/h/Java/Bonmin-1.8.6/SYMPHONY/lib64/pkgconfig:/cygdrive/h/Java/Bonmin-1.8.6/SYMPHONY/lib/pkgconfig:/cygdrive/h/Java/Bonmin-1.8.6/SYMPHONY/share/pkgconfig: i686-w64-mingw32-pkg-config --libs osi-sym`
  else
    ifeq ($(COIN_CXX_IS_CL), TRUE)
      LIBS = -link -libpath:`$(CYGPATH_W) /cygdrive/h/Java/Bonmin-1.8.6/SYMPHONY/lib` libOsiSym.lib libSym.lib 
    else
      LIBS = -L/cygdrive/h/Java/Bonmin-1.8.6/SYMPHONY/lib -lOsiSym -lSym 
    endif
  endif

  CXXFLAGS += -DUSE_OSI_INTERFACE

else
  ifeq ($(COIN_HAS_PKGCONFIG), TRUE)
    INCL = `PKG_CONFIG_PATH=/cygdrive/h/Java/Bonmin-1.8.6/SYMPHONY/lib64/pkgconfig:/cygdrive/h/Java/Bonmin-1.8.6/SYMPHONY/lib/pkgconfig:/cygdrive/h/Java/Bonmin-1.8.6/SYMPHONY/share/pkgconfig: i686-w64-mingw32-pkg-config --cflags symphony`
  else
    INCL = 
  endif

  ifeq ($(COIN_HAS_PKGCONFIG), TRUE)
    LIBS = `PKG_CONFIG_PATH=/cygdrive/h/Java/Bonmin-1.8.6/SYMPHONY/lib64/pkgconfig:/cygdrive/h/Java/Bonmin-1.8.6/SYMPHONY/lib/pkgconfig:/cygdrive/h/Java/Bonmin-1.8.6/SYMPHONY/share/pkgconfig: i686-w64-mingw32-pkg-config --libs symphony`
  else
    ifeq ($(COIN_CXX_IS_CL), TRUE)
      LIBS = -link -libpath:`$(CYGPATH_W) /cygdrive/h/Java/Bonmin-1.8.6/SYMPHONY/lib` libSym.lib 
    else
      LIBS = -L/cygdrive/h/Java/Bonmin-1.8.6/SYMPHONY/lib -lSym 
    endif
  endif
endif
INCL += $(ADDINCFLAGS)

all: $(WHATTOMAKE)

%.exe: %.c
	bla=;\
	for file in $< ; do bla="$$bla `$(CYGPATH_W) $$file`"; done; \
	$(CXX) $(CXXLINKFLAGS) $(CXXFLAGS) $(INCL) $$bla -o $@ $(LIBS)

clean:
	rm -rf $(WHATTOMAKE)
