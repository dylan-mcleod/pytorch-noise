
#ifndef BINDINGS_H
#define BINDINGS_H

#include <torch/types.h>

torch::Tensor eval_range3D(float startX, float startY, float startZ, float stepSizeX, float stepSizeY, float stepSizeZ, int stepX, int stepY, int stepZ);



torch::Tensor eval_simplexNoise_(torch::Tensor out, torch::Tensor points, float scale, long long seed);
torch::Tensor eval_checker_(torch::Tensor out, torch::Tensor points, float scale, long long seed);
torch::Tensor eval_spots_(torch::Tensor out, torch::Tensor points, float scale, long long seed, float size, int minNum, int maxNum, float jitter, int shape);
torch::Tensor eval_worleyNoise_(torch::Tensor out, torch::Tensor points, float scale, long long seed, int minNum, int maxNum, float jitter);
torch::Tensor eval_worleyNoise_five_(torch::Tensor out, torch::Tensor points, float scale, long long seed, int minNum, int maxNum, float jitter);
torch::Tensor eval_discreteNoise_(torch::Tensor out, torch::Tensor points, float scale, long long seed);
torch::Tensor eval_linearValue_(torch::Tensor out, torch::Tensor points, float scale, long long seed);
torch::Tensor eval_fadedValue_(torch::Tensor out, torch::Tensor points, float scale, long long seed);
torch::Tensor eval_cubicValue_(torch::Tensor out, torch::Tensor points, float scale, long long seed);
torch::Tensor eval_perlinNoise_(torch::Tensor out, torch::Tensor points, float scale, long long seed);
torch::Tensor eval_repeaterPerlin_(torch::Tensor out, torch::Tensor points, float scale, long long seed, int n, float lacunarity, float decay);
torch::Tensor eval_repeaterPerlinAbs_(torch::Tensor out, torch::Tensor points, float scale, long long seed, int n, float lacunarity, float decay);
torch::Tensor eval_repeaterSimplex_(torch::Tensor out, torch::Tensor points, float scale, long long seed, int n, float lacunarity, float decay);
torch::Tensor eval_repeaterSimplexAbs_(torch::Tensor out, torch::Tensor points, float scale, long long seed, int n, float lacunarity, float decay);
torch::Tensor eval_repeater_(torch::Tensor out, torch::Tensor points, float scale, long long seed, int n, float lacunarity, float decay, int basis);
torch::Tensor eval_fractalSimplex_(torch::Tensor out, torch::Tensor points, float scale, long long seed, float du, int n, float lacunarity, float decay);
torch::Tensor eval_turbulence_(torch::Tensor out, torch::Tensor points, float scaleIn, float scaleOut, long long seed, float strength, int inFunc, int outFunc);
torch::Tensor eval_repeaterTurbulence_(torch::Tensor out, torch::Tensor points, float scaleIn, float scaleOut, long long seed, float strength, int n, int inFunc, int outFunc);


#endif