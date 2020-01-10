

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
		torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> start, 
		float3 step_size,
		dim3 steps) {
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;
		const int z = blockIdx.z * blockDim.z + threadIdx.z;
		
	if (x < out.size(0) && y < out.size(1) && z < out.size(2)) {
		out[x][y][z][0] = start[0] + x*step_size.x;
		out[x][y][z][1] = start[1] + y*step_size.y;
		out[x][y][z][2] = start[2] + z*step_size.z;	
	}
}

torch::Tensor eval_range3D(
		torch::Tensor start, 
		float stepSizeX, float stepSizeY, float stepSizeZ, 
		int stepX, int stepY, int stepZ) {
	dim3 steps(stepX, stepY, stepZ);
	const dim3 threads(32,16,1);
	const dim3 blocks((steps.x+threads.x-1)/threads.x,(steps.y+threads.y-1)/threads.y,(steps.z+threads.z-1)/threads.z);
	
	auto out = torch::empty({steps.x,steps.y,steps.z,3}, at::kCUDA);
	
	AT_DISPATCH_FLOATING_TYPES(out.type(), "simplex3_forward_cuda", ([&] {
	
		kernel_range3D<scalar_t><<<blocks, threads>>>(
			out.      packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
			start.    packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
			make_float3(stepSizeX, stepSizeY, stepSizeZ),
			steps
		);
			
		cudaDeviceSynchronize();
	
	}));
	
	return out;
}
	

/*
template<typename scalar_t>
struct Float_Args {
	typedef torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> type;
};
typedef torch::PackedTensorAccessor<long long,1,torch::RestrictPtrTraits,size_t> Int_Args;

template<typename scalar_t>
__device__ __forceinline__
float call_simplexNoise(float3 pt, Float_Args<scalar_t>::type flt_args, Int_Args int_args) {
	return cudaNoise::simplexNoise(pt, flt_args[0], int_args[0]);
}

template<typename scalar_t>
__device__ __forceinline__
float call_checker(float3 pt, Float_Args<scalar_t>::type flt_args, Int_Args int_args) {
	return cudaNoise::checker(pt, flt_args[0], int_args[0]);
}

template<typename scalar_t>
__device__ __forceinline__
float call_spots(float3 pt, Float_Args<scalar_t>::type flt_args, Int_Args int_args) {
	return cudaNoise::spots(pt, flt_args[0], int_args[0], flt_args[1], int_args[1], int_args[2], flt_args[2], cudaNoise::profileShape(int_args[3]));
}

template<typename scalar_t>
__device__ __forceinline__
float call_worleyNoise(float3 pt, Float_Args<scalar_t>::type flt_args, Int_Args int_args) {
	return cudaNoise::worleyNoise(pt, flt_args[0], int_args[0], flt_args[1], int_args[1], int_args[2], flt_args[2]);
}

template<typename scalar_t>
__device__ __forceinline__
float call_discreteNoise(float3 pt, Float_Args<scalar_t>::type flt_args, Int_Args int_args) {
	return cudaNoise::discreteNoise(pt, flt_args[0], int_args[0]);
}

template<typename scalar_t>
__device__ __forceinline__
float call_linearValue(float3 pt, Float_Args<scalar_t>::type flt_args, Int_Args int_args) {
	return cudaNoise::linearValue(pt, flt_args[0], int_args[0]);
}

template<typename scalar_t>
__device__ __forceinline__
float call_fadedValue(float3 pt, Float_Args<scalar_t>::type flt_args, Int_Args int_args) {
	return cudaNoise::fadedValue(pt, flt_args[0], int_args[0]);
}

template<typename scalar_t>
__device__ __forceinline__
float call_cubicValue(float3 pt, Float_Args<scalar_t>::type flt_args, Int_Args int_args) {
	return cudaNoise::cubicValue(pt, flt_args[0], int_args[0]);
}

template<typename scalar_t>
__device__ __forceinline__
float call_perlinNoise(float3 pt, Float_Args<scalar_t>::type flt_args, Int_Args int_args) {
	return cudaNoise::perlinNoise(pt, flt_args[0], int_args[0]);
}

template<typename scalar_t>
__device__ __forceinline__
float call_repeaterPerlin(float3 pt, Float_Args<scalar_t>::type flt_args, Int_Args int_args) {
	return cudaNoise::repeaterPerlin(pt, flt_args[0], int_args[0], int_args[1], flt_args[1], flt_args[2]);
}

template<typename scalar_t>
__device__ __forceinline__
float call_repeaterPerlinAbs(float3 pt, Float_Args<scalar_t>::type flt_args, Int_Args int_args) {
	return cudaNoise::repeaterPerlinAbs(pt, flt_args[0], int_args[0], int_args[1], flt_args[1], flt_args[2]);
}

template<typename scalar_t>
__device__ __forceinline__
float call_repeaterSimplex(float3 pt, Float_Args<scalar_t>::type flt_args, Int_Args int_args) {
	return cudaNoise::repeaterSimplex(pt, flt_args[0], int_args[0], int_args[1], flt_args[1], flt_args[2]);
}

template<typename scalar_t>
__device__ __forceinline__
float call_repeaterSimplexAbs(float3 pt, Float_Args<scalar_t>::type flt_args, Int_Args int_args) {
	return cudaNoise::repeaterSimplexAbs(pt, flt_args[0], int_args[0], int_args[1], flt_args[1], flt_args[2]);
}

template<typename scalar_t>
__device__ __forceinline__
float call_repeater(float3 pt, Float_Args<scalar_t>::type flt_args, Int_Args int_args) {
	return cudaNoise::repeater(pt, flt_args[0], int_args[0], int_args[1], flt_args[1], flt_args[2], cudaNoise::basisFunction(int_args[2]));
}

template<typename scalar_t>
__device__ __forceinline__
float call_fractalSimplex(float3 pt, Float_Args<scalar_t>::type flt_args, Int_Args int_args) {
	return cudaNoise::fractalSimplex(pt, flt_args[0], int_args[0], flt_args[1], int_args[1], flt_args[2], flt_args[3]);
}

template<typename scalar_t>
__device__ __forceinline__
float call_turbulence(float3 pt, Float_Args<scalar_t>::type flt_args, Int_Args int_args) {
	return cudaNoise::turbulence(pt, flt_args[0], flt_args[1], int_args[0], flt_args[2], cudaNoise::basisFunction(int_args[1]), cudaNoise::basisFunction(int_args[2]));
}

template<typename scalar_t>
__device__ __forceinline__
float call_repeaterTurbulence(float3 pt, Float_Args<scalar_t>::type flt_args, Int_Args int_args) {
	return cudaNoise::repeaterTurbulence(pt, flt_args[0], flt_args[1], int_args[0], flt_args[2], int_args[1], cudaNoise::basisFunction(int_args[2]), cudaNoise::basisFunction(int_args[3]));
}




#define MAKE_KERNEL( NAME )\
template<typename scalar_t>\
__global__ void kernel_##NAME(\
		const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> points,\
		      torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> out,\
		Float_Args<scalar_t>::type flt_args,\
		Int_Args int_args) {\
	\
	const int x = blockIdx.x * blockDim.x + threadIdx.x;\
	const int y = blockIdx.y * blockDim.y + threadIdx.y;\
	const int z = blockIdx.z * blockDim.z + threadIdx.z;\
	if (x < points.size(0) && y < points.size(1) && z < points.size(2)) {\
		out[x][y][z] = call_##NAME <scalar_t>(make_float3(points[x][y][z][0],points[x][y][z][1],points[x][y][z][2]), flt_args, int_args);\
	}\
}

#define MAKE_EVAL( NAME )\
torch::Tensor eval_##NAME(torch::Tensor points, torch::Tensor flt_args, torch::Tensor int_args) {\
	\
	CHECK_INPUT(points);\
	\
	auto out = torch::empty({points.size(0),points.size(1),points.size(2)}, at::kCUDA);\
	\
	const dim3 threads(32,16,1);\
	const dim3 blocks((points.size(0)+threads.x-1)/threads.x,(points.size(1)+threads.y-1)/threads.y,(points.size(2)+threads.z-1)/threads.z);\
	\
	AT_DISPATCH_FLOATING_TYPES(points.type(), "simplex3_forward_cuda", ([&] {\
	\
		kernel_##NAME<scalar_t><<<blocks, threads>>>(\
			points.  packed_accessor<scalar_t, 4,torch::RestrictPtrTraits,size_t>(),\
			out.     packed_accessor<scalar_t, 3,torch::RestrictPtrTraits,size_t>(),\
			flt_args.packed_accessor<scalar_t, 1,torch::RestrictPtrTraits,size_t>(),\
			int_args.packed_accessor<long long,1,torch::RestrictPtrTraits,size_t>()\
		);\
	\
	}));\
	\
	return out;\
}

#define MAKE_EVAL_( NAME )\
torch::Tensor eval_##NAME##_(torch::Tensor out, torch::Tensor points, torch::Tensor flt_args, torch::Tensor int_args) {\
	\
	CHECK_INPUT(points);\
	CHECK_INPUT(out);\
	\
	const dim3 threads(32,16,1);\
	const dim3 blocks((points.size(0)+threads.x-1)/threads.x,(points.size(1)+threads.y-1)/threads.y,(points.size(2)+threads.z-1)/threads.z);\
	\
	AT_DISPATCH_FLOATING_TYPES(points.type(), "simplex3_forward_cuda", ([&] {\
	\
		kernel_##NAME<scalar_t><<<blocks, threads>>>(\
			points.  packed_accessor<scalar_t, 4,torch::RestrictPtrTraits,size_t>(),\
			out.     packed_accessor<scalar_t, 3,torch::RestrictPtrTraits,size_t>(),\
			flt_args.packed_accessor<scalar_t, 1,torch::RestrictPtrTraits,size_t>(),\
			int_args.packed_accessor<long long,1,torch::RestrictPtrTraits,size_t>()\
		);\
	\
	}));\
	\
	return out;\
}

#define MAKE_NOISE_FUN( NAME ) MAKE_KERNEL(NAME); MAKE_EVAL(NAME); MAKE_EVAL_(NAME);

MAKE_NOISE_FUN(simplexNoise)
MAKE_NOISE_FUN(checker)
MAKE_NOISE_FUN(spots)
MAKE_NOISE_FUN(worleyNoise)
MAKE_NOISE_FUN(discreteNoise)
MAKE_NOISE_FUN(linearValue)
MAKE_NOISE_FUN(fadedValue)
MAKE_NOISE_FUN(cubicValue)
MAKE_NOISE_FUN(perlinNoise)
MAKE_NOISE_FUN(repeaterPerlin)
MAKE_NOISE_FUN(repeaterPerlinAbs)
MAKE_NOISE_FUN(repeaterSimplex)
MAKE_NOISE_FUN(repeaterSimplexAbs)
MAKE_NOISE_FUN(repeater)
MAKE_NOISE_FUN(fractalSimplex)
MAKE_NOISE_FUN(turbulence)
MAKE_NOISE_FUN(repeaterTurbulence)
*/