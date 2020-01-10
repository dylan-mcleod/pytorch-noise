import torch
import numpy as np
import render_noise
from pytorch_noise import *


torch.set_default_tensor_type('torch.cuda.FloatTensor')
cuda = torch.device('cuda')
with torch.cuda.device(0):

    start =  torch.tensor([0.0,0.0,0.0]).cuda()
    update = torch.tensor([0.0,0.0,0.5]).cuda()
    
    noises = []
    
    """
    noises.append(Spots(0.04, 1234, 0.1, 1, 1, 1.0, 0))
    noises.append(WorleyNoise(0.04, 1234, 1, 1, 1.0))
    noises.append((WorleyNoise(0.04, 1234, 1, 1, 1.0) + Spots(0.04, 1234, 0.2, 1, 1, 1.0, 0)) * 0.5)
    """
    
    noises.append(octaveWorleyNoise(0.01, 1234, 1, 2.0, 0.6, minNum = 2, maxNum = 4, jitter = 1.0, searchFive=False))
    noises.append(octaveWorleyNoise(0.01, 1234, 10, 2.0, 0.6, minNum = 2, maxNum = 4, jitter = 1.0, searchFive=False))
    #noises.append(octaveWorleyNoise(0.01, 1234, 10, 2.0, 0.6, minNum = 1, maxNum = 1, jitter = 2.0, searchFive=True))
    
    """
    Test whether jitter = 1.0 has artefacts. Theoretically *some* artefacts should appear, it should be exceedingly unlikely. 
    (imagine one samples a point along an edge, and 2 grid spaces away, another point is as close to this one as possible)
    """
    """
    noises.append((octaveWorleyNoise(0.03, 1234, 1, 2.0, 0.6, minNum = 1, maxNum = 1, jitter = 1.0, searchFive=False)\
                - octaveWorleyNoise(0.03, 1234, 1, 2.0, 0.6, minNum = 1, maxNum = 1, jitter = 1.0, searchFive=True))*100.0)
    """
    
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
    noises.append(RepeaterSimplex(0.01, 98345, 5, 2.0, 0.5) * (octaveWorleyNoise(0.01, 1234, 3, 2.0, 0.6)*0.5 + 0.5))
    
    """
    Though the seeds will be (slightly) different, these two noises should have the same character.
    """
    noises.append(repeaterFactory(lambda seed,scale: PerlinNoise(scale, seed), 1234, 10, initial_params = 0.01, decayFunc = lambda x:x*0.5))
    noises.append(RepeaterPerlin(0.01, 1234, 5, 2.0, 0.5) * 2.0)
    
    """
    Multiplying perlin noise with itself, for the heck of it.
    """
    noises.append(RepeaterPerlin(0.01, 98345, 5, 2.0, 0.5) * RepeaterPerlin(0.01, 3548, 5, 2.0, 0.5) * 4.0)
    
    noises.append(PerlinNoise(0.02,1234))
    noises.append(SimplexNoise(0.1, 101234))
    
    
    render_noise.render_noise(noises, start, update, (1.01,1.01,1.01), (512,128))