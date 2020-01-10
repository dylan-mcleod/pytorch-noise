
#ifndef BINDINGS_H
#define BINDINGS_H

#include <torch/types.h>

torch::Tensor eval_range3D(torch::Tensor start, float stepSizeX, float stepSizeY, float stepSizeZ, int stepX, int stepY, int stepZ);

/*
#define DECL_NOISE_FUN(NAME)\
torch::Tensor eval_##NAME(torch::Tensor points, torch::Tensor flt_args, torch::Tensor int_args);\
torch::Tensor eval_##NAME##_(torch::Tensor out, torch::Tensor points, torch::Tensor flt_args, torch::Tensor int_args);

DECL_NOISE_FUN(simplexNoise)
DECL_NOISE_FUN(checker)
DECL_NOISE_FUN(spots)
DECL_NOISE_FUN(worleyNoise)
DECL_NOISE_FUN(discreteNoise)
DECL_NOISE_FUN(linearValue)
DECL_NOISE_FUN(fadedValue)
DECL_NOISE_FUN(cubicValue)
DECL_NOISE_FUN(perlinNoise)
DECL_NOISE_FUN(repeaterPerlin)
DECL_NOISE_FUN(repeaterPerlinAbs)
DECL_NOISE_FUN(repeaterSimplex)
DECL_NOISE_FUN(repeaterSimplexAbs)
DECL_NOISE_FUN(repeater)
DECL_NOISE_FUN(fractalSimplex)
DECL_NOISE_FUN(turbulence)
DECL_NOISE_FUN(repeaterTurbulence)*/



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