
#include <torch/script.h>
#include <torch/python.h>
#include "bindings.h"

#define DEF_NOISE_BINDING( NAME, DESC )\
m.def(#NAME "_", &eval_##NAME##_, DESC)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	
	m.def("range3D",&eval_range3D,"");
	DEF_NOISE_BINDING(simplexNoise, "3D simplex noise");
	DEF_NOISE_BINDING(checker, "3D checkerboard noise");
	DEF_NOISE_BINDING(spots, "");
	DEF_NOISE_BINDING(worleyNoise, "");
	DEF_NOISE_BINDING(worleyNoise_five, "");
	DEF_NOISE_BINDING(discreteNoise, "");
	DEF_NOISE_BINDING(linearValue, "");
	DEF_NOISE_BINDING(fadedValue, "");
	DEF_NOISE_BINDING(cubicValue, "");
	DEF_NOISE_BINDING(perlinNoise, "");
	DEF_NOISE_BINDING(repeaterPerlin, "");
	DEF_NOISE_BINDING(repeaterPerlinAbs, "");
	DEF_NOISE_BINDING(repeaterSimplex, "");
	DEF_NOISE_BINDING(repeaterSimplexAbs, "");
	DEF_NOISE_BINDING(repeater, "");
	DEF_NOISE_BINDING(fractalSimplex, "");
	DEF_NOISE_BINDING(turbulence, "");
	DEF_NOISE_BINDING(repeaterTurbulence, "");
}