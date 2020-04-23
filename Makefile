CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
# NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
NVCC = nvcc
# select one of these for Debug vs. Release
#NVCC_DBG       = -g -G
NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64 -G
GENCODE_FLAGS  = -gencode arch=compute_50,code=sm_50

SRCS = main.cu
INCS = Vector3.h Ray.h VisibleObject.h World.h Sphere.h Plane.h RenderEngine.h Managed.h TriangularMesh.h

render: render.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o render render.o

render.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o render.o -c main.cu

out.ppm: render
	rm -f out.ppm
	./render > out.ppm

out.jpg: out.ppm
	rm -f out.jpg
	ppmtojpeg out.ppm > out.jpg

profile_basic: render
	nvprof ./render > out.ppm

# use nvprof --query-metrics
profile_metrics: render
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./render > out.ppm

clean:
	rm -f render render.o out.ppm out.jpg
