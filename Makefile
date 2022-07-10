CC=gcc
CXX=g++
RM=rm -f
CPPFLAGS +=  -g  -fPIC -O2
LDFLAGS +=  -I /usr/include/python3.10
LDLIBS += -L /usr/lib/x86_64-linux-gnu -l:libboost_python310.so -l boost_numpy310 -l armadillo

TMPSRCS=$(wildcard Source_cpp/*.cpp)
SRCS = $(filter-out Source_cpp/binmixtC.cpp, $(TMPSRCS))
OBJS=$(subst .cpp,.o,$(SRCS))

LIB = binmixtC.so

all: lib

lib: $(OBJS)
	$(CXX) -shared $(CPPFLAGS) $(LDFLAGS) -o $(LIB) Source_cpp/binmixtC.cpp $(OBJS) $(LDLIBS)

%.o: %.cpp %.h
	$(CXX) $(CPPFLAGS) $(LDFLAGS) $(LDLIBS) -c $< -o $@

clean:
	$(RM) $(LIB) $(OBJS)
