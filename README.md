# DAIBench

DAIBench (`DiDi AI Benchmarks`) aims to provide a set of AI evaluation sets for production environments, spanning different types of GPU servers and cloud environments, to provide users with effective and credible test results for future hardware selection , software and library optimization, business model improvement, link stress testing and other stages to lay a solid data foundation and technical reference.

## Supported Features
- Layerwised benchmarking, from hardwares(L1), operators(L2) to models(L3), higher level benchmarking is TBD.
- Cloud-native benchmarking, containerized deploying, easy to use.
- Multi-cloud benchmarking, results are useful for price/performance considerations.


## General Structure

DAIBench comprehensively considers the existing GPU performance testing tools, and divides the indicators into hardware layer, framework (operator) layer, and algorithm layer.

For each level, DAIBench currently supports the following tests:

| Layer | Supported Test |
|:--------:|------|
|Hardware layer|Focusing on the indicators of the hardware itself, such as peak computing throughput (TFLOPS/TOPS) calculation indicators and memory access bandwidth, PCIe communication bandwidth and other I/O indicators.|
|Frame/operator layer|Evaluating the computing power of commonly used operators (convolution, Softmax, matrix multiplication, etc.) based on mainstream AI frameworks.
|Model layer|Performing  end-to-end evaluation by selecting models in a series of production tasks.|

## Getting started
### Hardware Layer
```
cd <test_folder>
bash install.sh
bash run.sh
```

For GPU test, please install suitable `nvidia-driver` and `cuda` first.

### Operator Layer
Current operator layer is using [DeepBench](https://github.com/baidu-research/DeepBench)

```
cd operator
bash install.sh # download source code & prepare nccl
```

To run GEMM, convolution, recurrent op and sparse GEMM benchmarks:

```
bin/gemm_bench <inference|train> <int8|float|half>
```

To execute the NCCL single All-Reduce benchmark:

```
bin/nccl_single_all_reduce <num_gpus>
```

The NCCL MPI All-Reduce benchmark can be run using mpirun as shown below:

```
mpirun -np <num_ranks> bin/nccl_mpi_all_reduce
```
num_ranks cannot be greater than the number of GPUs in the system.

### Model layer

`docker` and `nvidia-docker` is required for model testing. To run specific model, please read `Readme.md` in the folder.

General test procedure:

- Download dataset
- Preprocess dataset (if needed)
- Build docker
- Launch benchmark
- Get result

## Developer guide
See `wiki` for guidelines.

## Contributing
Welcome to contribute by creating issues or sending pull requests. See `Contributing Guide` for guidelines.

## License
DAIBench is licensed under the `Apache License 2.0`.
