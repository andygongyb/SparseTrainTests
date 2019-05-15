SOURCES := $(wildcard *.cpp)
VERIFY_TARGETS := $(SOURCES:.cpp=_verify)
NOVERIFY_TARGETS := $(SOURCES:.cpp=_noverify)
MKLDNN=/andy/mkl-dnn-v0.90
VTUNE=/opt/intel-2019.0/vtune_amplifier_2019.0.2.570779


all: $(VERIFY_TARGETS) $(NOVERIFY_TARGETS)

%_verify: %.cpp
	icpc -O3 -g -std=c++11 -qopenmp $< -o $@ -DVERIFY=true $(VTUNE)/lib64/libittnotify.a -I$(MKLDNN)/include -I$(MKLDNN)/tests/gtests -I$(MKLDNN) /andy/packages/lib/libmkldnn.so


%_noverify: %.cpp
	icpc -O3 -g -std=c++11 -qopenmp $< -o $@ -DVERIFY=false $(VTUNE)/lib64/libittnotify.a -I$(MKLDNN)/include -I$(MKLDNN)/tests/gtests -I$(MKLDNN) /andy/packages/lib/libmkldnn.so

