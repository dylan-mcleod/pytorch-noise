

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/cuda.h>
#include "cuda_noise.cuh"
#include "bindings.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

	

using namespace cudaNoise;

#define MAKE_KERNEL_FUN(noiseFunc)\
template<typename scalar_t, typename... Args>\
__global__ void noiseFunc##_kernel (\
		const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> points,\
		      torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> out,\
			  Args... args) {\
	\
	const int x = blockIdx.x * blockDim.x + threadIdx.x;\
	if (x < points.size(0)) {\
		out[x] = noiseFunc (make_float3(points[x][0],points[x][1],points[x][2]), args...);\
	}\
}

MAKE_KERNEL_FUN(simplexNoise)
MAKE_KERNEL_FUN(checker)
MAKE_KERNEL_FUN(spots)
MAKE_KERNEL_FUN(worleyNoise)
MAKE_KERNEL_FUN(worleyNoise_five)
MAKE_KERNEL_FUN(discreteNoise)
MAKE_KERNEL_FUN(linearValue)
MAKE_KERNEL_FUN(fadedValue)
MAKE_KERNEL_FUN(cubicValue)
MAKE_KERNEL_FUN(perlinNoise)
MAKE_KERNEL_FUN(repeaterPerlin)
MAKE_KERNEL_FUN(repeaterPerlinAbs)
MAKE_KERNEL_FUN(repeaterSimplex)
MAKE_KERNEL_FUN(repeaterSimplexAbs)
MAKE_KERNEL_FUN(repeater)
MAKE_KERNEL_FUN(fractalSimplex)
MAKE_KERNEL_FUN(turbulence)
MAKE_KERNEL_FUN(repeaterTurbulence)




#define evalPoints(noiseFunc, points, ...)\
CHECK_INPUT(points);\
auto out = torch::empty({points.size(0)}, at::kCUDA);\
\
const int threads = 1024;\
const int blocks = (points.size(0)+threads-1)/threads;\
\
AT_DISPATCH_FLOATING_TYPES(points.type(), "noise", ([&] {\
	\
	noiseFunc##_kernel<scalar_t><<<blocks, threads>>>(\
		points.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),\
		out.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(), __VA_ARGS__);\
\
}));\
\
return out;



#define evalPoints_(noiseFunc, out, points, ...)\
CHECK_INPUT(points);\
CHECK_INPUT(out);\
\
const int threads = 1024;\
const int blocks = (points.size(0)+threads-1)/threads;\
\
AT_DISPATCH_FLOATING_TYPES(points.type(), "noise", ([&] {\
	\
	noiseFunc##_kernel<scalar_t><<<blocks, threads>>>(\
		points.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),\
		out.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(), __VA_ARGS__);\
	\
}));\
\
return out;




torch::Tensor eval_simplexNoise_(torch::Tensor out, torch::Tensor points, float scale, long long seed) {
	evalPoints_(simplexNoise, out, points, scale, (int) seed);
}

torch::Tensor eval_checker_(torch::Tensor out, torch::Tensor points, float scale, long long seed) {
	evalPoints_(checker, out, points, scale, (int) seed);
}

torch::Tensor eval_spots_(torch::Tensor out, torch::Tensor points, float scale, long long seed, float size, int minNum, int maxNum, float jitter, int shape) {
	evalPoints_(spots, out, points, scale, (int) seed, size, minNum, maxNum, jitter, profileShape(shape));
}

torch::Tensor eval_worleyNoise_(torch::Tensor out, torch::Tensor points, float scale, long long seed, int minNum, int maxNum, float jitter) {
	evalPoints_(worleyNoise, out, points, scale, (int) seed, minNum, maxNum, jitter);
}

torch::Tensor eval_worleyNoise_five_(torch::Tensor out, torch::Tensor points, float scale, long long seed, int minNum, int maxNum, float jitter) {
	evalPoints_(worleyNoise_five, out, points, scale, (int) seed, minNum, maxNum, jitter);
}

torch::Tensor eval_discreteNoise_(torch::Tensor out, torch::Tensor points, float scale, long long seed) {
	evalPoints_(discreteNoise, out, points, scale, (int) seed);
}

torch::Tensor eval_linearValue_(torch::Tensor out, torch::Tensor points, float scale, long long seed) {
	evalPoints_(linearValue, out, points, scale, (int) seed);
}

torch::Tensor eval_fadedValue_(torch::Tensor out, torch::Tensor points, float scale, long long seed) {
	evalPoints_(fadedValue, out, points, scale, (int) seed);
}

torch::Tensor eval_cubicValue_(torch::Tensor out, torch::Tensor points, float scale, long long seed) {
	evalPoints_(cubicValue, out, points, scale, (int) seed);
}

torch::Tensor eval_perlinNoise_(torch::Tensor out, torch::Tensor points, float scale, long long seed) {
	evalPoints_(perlinNoise, out, points, scale, (int) seed);
}

torch::Tensor eval_repeaterPerlin_(torch::Tensor out, torch::Tensor points, float scale, long long seed, int n, float lacunarity, float decay) {
	evalPoints_(repeaterPerlin, out, points, scale, (int) seed, n, lacunarity, decay);
}

torch::Tensor eval_repeaterPerlinAbs_(torch::Tensor out, torch::Tensor points, float scale, long long seed, int n, float lacunarity, float decay) {
	evalPoints_(repeaterPerlinAbs, out, points, scale, (int) seed, n, lacunarity, decay);
}

torch::Tensor eval_repeaterSimplex_(torch::Tensor out, torch::Tensor points, float scale, long long seed, int n, float lacunarity, float decay) {
	evalPoints_(repeaterSimplex, out, points, scale, (int) seed, n, lacunarity, decay);
}

torch::Tensor eval_repeaterSimplexAbs_(torch::Tensor out, torch::Tensor points, float scale, long long seed, int n, float lacunarity, float decay) {
	evalPoints_(repeaterSimplexAbs, out, points, scale, (int) seed, n, lacunarity, decay);
}

torch::Tensor eval_repeater_(torch::Tensor out, torch::Tensor points, float scale, long long seed, int n, float lacunarity, float decay, int basis) {
	evalPoints_(repeater, out, points, scale, (int) seed, n, lacunarity, decay, basisFunction(basis));
}

torch::Tensor eval_fractalSimplex_(torch::Tensor out, torch::Tensor points, float scale, long long seed, float du, int n, float lacunarity, float decay) {
	evalPoints_(fractalSimplex, out, points, scale, (int) seed, du, n, lacunarity, decay);
}

torch::Tensor eval_turbulence_(torch::Tensor out, torch::Tensor points, float scaleIn, float scaleOut, long long seed, float strength, int inFunc, int outFunc) {
	evalPoints_(turbulence, out, points, scaleIn, scaleOut, (int) seed, strength, basisFunction(inFunc), basisFunction(outFunc));
}

torch::Tensor eval_repeaterTurbulence_(torch::Tensor out, torch::Tensor points, float scaleIn, float scaleOut, long long seed, float strength, int n, int inFunc, int outFunc) {
	evalPoints_(repeaterTurbulence, out, points, scaleIn, scaleOut, (int) seed, strength, n, basisFunction(inFunc), basisFunction(outFunc));
}




template<typename scalar_t>
__global__
void kernel_range3D(
		torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> out,
		float3 start, 
		float3 step_size,
		dim3   steps) {
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;
		const int z = blockIdx.z * blockDim.z + threadIdx.z;
		
	if (x < out.size(0) && y < out.size(1) && z < out.size(2)) {
		out[x][y][z][0] = start.x + x*step_size.x;
		out[x][y][z][1] = start.y + y*step_size.y;
		out[x][y][z][2] = start.z + z*step_size.z;	
	}
}

torch::Tensor eval_range3D(
		float startX, float startY, float startZ, 
		float stepSizeX, float stepSizeY, float stepSizeZ, 
		int stepX, int stepY, int stepZ) {
	dim3 steps(stepX, stepY, stepZ);
	const dim3 threads(32,16,1);
	const dim3 blocks((steps.x+threads.x-1)/threads.x,(steps.y+threads.y-1)/threads.y,(steps.z+threads.z-1)/threads.z);
	
	auto out = torch::empty({steps.x,steps.y,steps.z,3}, at::kCUDA);
	
	AT_DISPATCH_FLOATING_TYPES(out.type(), "simplex3_forward_cuda", ([&] {
	
		kernel_range3D<scalar_t><<<blocks, threads>>>(
			out.      packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
			make_float3(startX,    startY,    startZ   ),
			make_float3(stepSizeX, stepSizeY, stepSizeZ),
			steps
		);
			
		cudaDeviceSynchronize();
	
	}));
	
	return out;
}