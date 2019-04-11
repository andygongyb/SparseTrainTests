
# Validation for SparseTrain

This repository contains testers for SparseTrain. Please modify the Makefile to suit your environment. Compiling the testers requires SparseTrain and Intel VTune Amplifier being installed. You can run the testers with or without VTune. After running `make`, you will get 2 executables from each source file. The one with `_verify` suffix verifies the output from SparseTrain against a (very slow) reference convolution. Because the verification takes very long time, we suggest you only verify with small convolution configureations. For example, N=16 C=128 W=7 H=7 K=128 O=1 P=1 R=3 S=3. The one with `_noverify` as suffix does not perform such verification. The executables to test SparseTrain are:

```
./fwd_trans_test_verify
./fwd_trans_test_noverify
./bwd_trans_test_verify
./bwd_trans_test_noverify
./bww_trans_test_verify
./bww_trans_test_noverify
```
The others are for internal tests. `fwd` is forward propagation; `bwd` is backward propagation by input; `bww` is backward propagation by weights. To execute the tests, please provide the arguments as:

```
./fwd_trans_test_noverify [sparsity percentage] [iterations] [batch size] [input channel] [input width] [input height] [output channel] [horizontal stride] [vertical stride] [filter width] [filter height]
```
For example
```
./fwd_trans_test_noverify 70 10000 16 256 14 14 512 1 1 3 3
```
runs the forward propagation at 70% sparsity for 10000 iterations. The convolution size is set to N=16 C=256 W=14 H=14 K=512 O=1 P=1 R=3 S=3.

The testing programs output the execution time in cycles measured with the RDTSCP instruction to standard out.
