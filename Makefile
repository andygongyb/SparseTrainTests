SOURCES := $(wildcard *.cpp)
VERIFY_TARGETS := $(SOURCES:.cpp=_verify)
NOVERIFY_TARGETS := $(SOURCES:.cpp=_noverify)


all: $(VERIFY_TARGETS) $(NOVERIFY_TARGETS)

%_verify: %.cpp
	icpc -O3 -g -std=c++11 $< -o $@ -DVERIFY=true /andy/mkl-dnn-reorder-skip-trans-3/debug/tests/gtests/gtest/libmkldnn_gtest.a -lmkldnn /opt/intel-2019.0/vtune_amplifier_2019.0.2.570779/lib64/libittnotify.a -I/andy/mkl-dnn-reorder-skip-trans-3/include -I/andy/mkl-dnn-reorder-skip-trans-3/tests/gtests -I/andy/mkl-dnn-reorder-skip-trans-3 -lpthread


%_noverify: %.cpp
	icpc -O3 -g -std=c++11 $< -o $@ -DVERIFY=false /andy/mkl-dnn-reorder-skip-trans-3/debug/tests/gtests/gtest/libmkldnn_gtest.a -lmkldnn /opt/intel-2019.0/vtune_amplifier_2019.0.2.570779/lib64/libittnotify.a -I/andy/mkl-dnn-reorder-skip-trans-3/include -I/andy/mkl-dnn-reorder-skip-trans-3/tests/gtests -I/andy/mkl-dnn-reorder-skip-trans-3 -lpthread

