#------------------------------------------------------------------------------#
# This makefile was generated by 'cbp2make' tool rev.147                       #
#------------------------------------------------------------------------------#


WORKDIR = `pwd`

CC = gcc
CXX = g++
AR = ar
LD = g++
WINDRES = windres

INC = -I/usr/include/python2.7 -I.. -I../../../dlib -I../..
CFLAGS = -std=c++11 -Wall -fexceptions -fPIC -pthread -DARMA_DONT_USE_WRAPPER
RESINC = 
LIBDIR = -L/usr/lib/python2.7 -L/usr/local/lib64
LIB = -lenkicore -lblas -llapack
LDFLAGS = -pthread

INC_DEBUG = $(INC)
CFLAGS_DEBUG = $(CFLAGS) -g
RESINC_DEBUG = $(RESINC)
RCFLAGS_DEBUG = $(RCFLAGS)
LIBDIR_DEBUG = $(LIBDIR) -L../../bin/Debug
LIB_DEBUG = $(LIB)
LDFLAGS_DEBUG = $(LDFLAGS)
OBJDIR_DEBUG = obj/Debug
DEP_DEBUG = 
OUT_DEBUG = ../../enki/_api.so

INC_RELEASE = $(INC)
CFLAGS_RELEASE = $(CFLAGS) -O2
RESINC_RELEASE = $(RESINC)
RCFLAGS_RELEASE = $(RCFLAGS)
LIBDIR_RELEASE = $(LIBDIR) -L../../bin/Release
LIB_RELEASE = $(LIB)
LDFLAGS_RELEASE = $(LDFLAGS) -s
OBJDIR_RELEASE = obj/Release
DEP_RELEASE = 
OUT_RELEASE = ../../enki/_api.so

OBJ_DEBUG = $(OBJDIR_DEBUG)/api_wrap.o

OBJ_RELEASE = $(OBJDIR_RELEASE)/api_wrap.o

all: before_build build_debug build_release after_build

clean: clean_debug clean_release

before_build: 
	make -f Makefile.swig_run

after_build: 

before_debug: 
	test -d ../../enki || mkdir -p ../../enki
	test -d $(OBJDIR_DEBUG) || mkdir -p $(OBJDIR_DEBUG)

after_debug: 

build_debug: before_debug out_debug after_debug

debug: before_build build_debug after_build

out_debug: before_debug $(OBJ_DEBUG) $(DEP_DEBUG)
	$(LD) -shared $(LIBDIR_DEBUG) $(OBJ_DEBUG)  -o $(OUT_DEBUG) $(LDFLAGS_DEBUG) $(LIB_DEBUG)

$(OBJDIR_DEBUG)/api_wrap.o: api_wrap.cxx
	$(CXX) $(CFLAGS_DEBUG) $(INC_DEBUG) -c api_wrap.cxx -o $(OBJDIR_DEBUG)/api_wrap.o

clean_debug: 
	rm -f $(OBJ_DEBUG) $(OUT_DEBUG)
	rm -rf ../../enki
	rm -rf $(OBJDIR_DEBUG)

before_release: 
	test -d ../../enki || mkdir -p ../../enki
	test -d $(OBJDIR_RELEASE) || mkdir -p $(OBJDIR_RELEASE)

after_release: 

build_release: before_release out_release after_release

release: before_build build_release after_build

out_release: before_release $(OBJ_RELEASE) $(DEP_RELEASE)
	$(LD) -shared $(LIBDIR_RELEASE) $(OBJ_RELEASE)  -o $(OUT_RELEASE) $(LDFLAGS_RELEASE) $(LIB_RELEASE)

$(OBJDIR_RELEASE)/api_wrap.o: api_wrap.cxx
	$(CXX) $(CFLAGS_RELEASE) $(INC_RELEASE) -c api_wrap.cxx -o $(OBJDIR_RELEASE)/api_wrap.o

clean_release: 
	rm -f $(OBJ_RELEASE) $(OUT_RELEASE)
	rm -rf ../../enki
	rm -rf $(OBJDIR_RELEASE)

.PHONY: before_build after_build before_debug after_debug clean_debug before_release after_release clean_release

