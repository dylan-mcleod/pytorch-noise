import torch
import numpy as np
import render_noise
from pytorch_noise import *

class RandN_Noise(NoiseFun):
    """
    Not actually a well-defined noise function. It's just here for benchmarking against
    """
    
    def __init__(self, seed):
        self.seed = seed
    
    def forward_(self, out, points):
        torch.manual_seed(self.seed)
        self.seed += 1
        return torch.randn(out.size(), out=out)



with torch.cuda.device(0):
    
    
    start     = np.array([0.0, 0.0, 0.0])
    update    = np.array([0.0, 0.0, 0.5])
    
    step_size = np.array([1.0, 1.0])
    steps     = np.array([512, 128])
    
    noises = []
    
    """
    noises.append(Spots(0.04, 1234, 0.1, 1, 1, 1.0, 0))
    noises.append(WorleyNoise(0.04, 1234, 1, 1, 1.0))
    noises.append((WorleyNoise(0.04, 1234, 1, 1, 1.0) + Spots(0.04, 1234, 0.2, 1, 1, 1.0, 0)) * 0.5)
    """
    
    noises.append(octaveWorleyNoise(0.01, 1234, 1, 2.0, 0.6, minNum = 2, maxNum = 3, jitter = 1.0, searchFive=False))
    noises.append(Shifted(\
        octaveWorleyNoise(0.01, 1234, 5, 2.0, 0.35, minNum = 2, maxNum = 3, jitter = 1.0, searchFive=False),\
        np.array([0, step_size[1] * steps[1], 0])
    ))
    noises.append(Shifted(\
        octaveWorleyNoise(0.01, 1234, 5, 2.0, 0.55, minNum = 2, maxNum = 3, jitter = 1.0, searchFive=False),\
        np.array([0, step_size[1] * steps[1] * 2, 0])
    ))
    #noises.append(octaveWorleyNoise(0.01, 1234, 10, 2.0, 0.6, minNum = 1, maxNum = 1, jitter = 2.0, searchFive=True))
    
    """
    Test whether jitter = 1.0 has artefacts. Theoretically *some* artefacts should appear, it should be exceedingly unlikely. 
    (imagine one samples a point along an edge, and 2 grid spaces away, another point is as close to this one as possible)
    """
    '''
    noises.append((octaveWorleyNoise(0.03, 1234, 1, 2.0, 0.6, minNum = 1, maxNum = 1, jitter = 1.0, searchFive=False)\
                - octaveWorleyNoise(0.03, 1234, 1, 2.0, 0.6, minNum = 1, maxNum = 1, jitter = 1.0, searchFive=True))*100.0)
    '''
    
    """
    Though the seeds will be different, these two noises should have the same character.
    """
    """
    noises.append(repeaterFactory(lambda seed,scale: SimplexNoise(scale, seed), 1234, 10, initial_params = 0.01, decayFunc = lambda x:x*0.5))
    noises.append(RepeaterSimplex(0.01, 1234, 5, 2.0, 0.5))
    """
    
    """
    Simplex * Worley, why not?
    Though the seeds will be different, these two noises should have the same character.
    """
    noises.append(RepeaterPerlin(0.01, 98345, 5, 2.0, 0.5) / RepeaterPerlin(0.01, 23451, 5, 2.0, 0.5))
    noises.append(RepeaterSimplex(0.01, 98345, 5, 2.0, 0.5) * (octaveWorleyNoise(0.01, 1234, 1, 2.0, 0.6, minNum = 2, maxNum = 4, jitter = 1.0, searchFive=False) + 1.0) * 0.5)
    #noises.append(RandN_Noise(124565))
    
    """
    Though the seeds will be (slightly) different, these two noises should have the same character.
    """
    noises.append(repeaterFactory(lambda seed,scale: PerlinNoise(scale, seed), 1234, 10, initial_params = 0.01, decayFunc = lambda x:x*0.5))
    #noises.append(RepeaterPerlin(0.01, 1234, 5, 2.0, 0.5) * RepeaterPerlin(0.01, 15645, 5, 2.0, 0.5) * RepeaterPerlin(0.01, 3265, 5, 2.0, 0.5) * RepeaterPerlin(0.01, 12349, 5, 2.0, 0.5) * RepeaterPerlin(0.01, 19234, 5, 2.0, 0.5) * 32.0)
    
    """
    Multiplying perlin noise with itself, for the heck of it.
    """
    noises.append(RepeaterPerlin(0.01, 98345, 5, 2.0, 0.5) * RepeaterPerlin(0.01, 3548, 5, 2.0, 0.5) * 4.0)
    
    noises.append(PerlinNoise(0.02,1234))
    noises.append(SimplexNoise(0.1, 101234))
    
    render_noise.render_noise(noises, start, update, (1.0,1.0,1.0), (512,128), cuda_device=0)