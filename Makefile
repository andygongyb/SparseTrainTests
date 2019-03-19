SOURCES := $(wildcard *.cpp)
VERIFY_TARGETS := $(SOURCES:.cpp=_verify)
NOVERIFY_TARGETS := $(SOURCES:.cpp=_noverify)
MKLDNN=/andy/mkl-dnn-reorder-skip-trans-3
VTUNE=/opt/intel-2019.0/vtune_amplifier_2019.0.2.570779


all: $(VERIFY_TARGETS) $(NOVERIFY_TARGETS)

%_verify: %.cpp
	icpc -O3 -g -std=c++11 $< -o $@ -DVERIFY=true $(MKLDNN)/debug/tests/gtests/gtest/libmkldnn_gtest.a -lmkldnn $(VTUNE)/lib64/libittnotify.a -I$(MKLDNN)/include -I$(MKLDNN)/tests/gtests -I$(MKLDNN) -lpthread


%_noverify: %.cpp
	icpc -O3 -g -std=c++11 $< -o $@ -DVERIFY=false $(MKLDNN)/debug/tests/gtests/gtest/libmkldnn_gtest.a -lmkldnn $(VTUNE)/lib64/libittnotify.a -I$(MKLDNN)/include -I$(MKLDNN)/tests/gtests -I$(MKLDNN) -lpthread

