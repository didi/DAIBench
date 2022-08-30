all:
	/usr/local/cuda/bin/nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcublas -o calc_gpu_peak_gflops.bin calc_gpu_peak_gflops.cc

clean:
	rm -f *.bin
