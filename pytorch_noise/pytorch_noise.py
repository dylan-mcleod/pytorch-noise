import torch
import pytorch_noise_cuda as noise_lib
   
  
def range3D(start, step_size, steps):
    return noise_lib.range3D(start[0], start[1], start[2], step_size[0], step_size[1], step_size[2], steps[0], steps[1], steps[2])
            
            

def hashInt(seed):
    seed = (seed + 0x7ed55d16) + (seed << 12)
    seed = (seed ^ 0xc761c23c) ^ (seed >> 19)
    seed = (seed + 0x165667b1) + (seed << 5)
    seed = (seed + 0xd3a2646c) ^ (seed << 9)
    seed = (seed + 0xfd7046c5) + (seed << 3)
    seed = (seed ^ 0xb55a4f09) ^ (seed >> 16)
    return seed % (1<<63)
    
def hashInts(*vals):
    ret = 0
    for v in vals:
        ret *= 31
        ret ^= hashInt(v)
        ret %= 1<<63
    return ret
    
class NoiseFun:

    @property
    def amp(self):
        return 1.0
        
    @property
    def dim(self):
        return 3
        
    @property
    def channels(self):
        return 1
        
    def advanceSeed(self):
        if hasattr(self, 'seed'):
            self.seed = hashInt(self.seed)
        
    
        
        
    # TODO: Finish exception handling, and maybe change how 1-D noise is handled. That is, if anyone even cares about 1-D noise.
    # Note that out and points must be contiguous tensors. Thanks <3
    def forward(self, points, out=None, cuda_device=0):
        torch.cuda.init()
        orig_shape = ()
        if out is not None:
            orig_shape = out.size()
        else:
            if self.channels == 1:
                orig_shape = points.size()[:-1]
            else:
                orig_shape = (points.size()[:-1],self.channels)
        
        if points.dim() > 2:
            points = points.reshape((-1, self.dim))
            
        if points.dim() == 1:
            if self.dim == 1:
                points = points[..., None]
            else:
                raise Exception("Input tensor must be at least 2 dimensional")
    
        if points.size()[1] != self.dim:
            raise Exception("TODO message")
            
        if out is None:
            
            if points.size()[1] != self.dim:
                if self.dim != 1:
                    raise Exception("Input points do not have the same dimension as the noise function")
        
            if self.channels > 1:
                shape = (points.size()[0], self.channels)
            else:
                shape = (points.size()[0])
                
            out = torch.empty(shape, device = "cuda:"+str(cuda_device))
        else:
            out = out.reshape[..., self.channels]
            
            
        if out.size()[0] != points.size()[0]:
            raise Exception("Output tensor must have the same length as the input tensor")
            
        ret = self.forward_(out, points).reshape(orig_shape)
        torch.cuda.synchronize()
        return ret
        
    
    
    def __call__(self, points, out=None, cuda_device=0):
        return self.forward(points, out)
        
    # TODO: implement operator overloads for + - * / // % **
    # supporting both (NoiseFun, NoiseFun) and (NoiseFun, float)
    
    def __neg__(self):
        return Amplifier(self, -1.0)
    
    def __pos__(self):
        return self
        
    def __add__(self, other):
        if isinstance(other, NoiseFun):
            return Adder([self, other])
        return ScalarAdder(self, other)
        
    def __sub__(self, other):
        if isinstance(other, NoiseFun):
            return BinarySubtractor(self, other)
        return ScalarAdder(self, -other)
    
    def __mul__(self, other):
        if isinstance(other, NoiseFun):
            return BinaryOperator(self, other, lambda x,y: x.mul_(y))
        return Amplifier(self, other)
     
    def __truediv__(self, other):
        if isinstance(other, NoiseFun):
            return BinaryOperator(self, other, lambda x,y: x.div_(y))
        return Amplifier(self, 1.0/other)





class SimplexNoise(NoiseFun):
    
    def __init__(self, scale, seed):
        self.scale = scale
        self.seed = seed

    def forward_(self, out, points):
        return noise_lib.simplexNoise_(out, points, self.scale, self.seed)
        
        
class PerlinNoise(NoiseFun):
    def __init__(self, scale, seed):
        self.scale = scale
        self.seed = seed

    def forward_(self, out, points):
        return noise_lib.perlinNoise_(out, points, self.scale, self.seed)
        
class Spots(NoiseFun):
    def __init__(self, scale, seed, size, minNum, maxNum, jitter, profileShape):
        """
        """
        self.scale, self.seed, self.size, self.minNum, self.maxNum, self.jitter, self.profileShape =\
            scale, seed, size, minNum, maxNum, jitter, profileShape
            
    def forward_(self, out, points):
        return noise_lib.spots_(out, points, self.scale, self.seed, self.size, self.minNum, self.maxNum, self.jitter, self.profileShape)
        
        
class WorleyNoise(NoiseFun):
    def __init__(self, scale, seed, minNum, maxNum, jitter, searchFive = False):
        self.scale = scale
        self.seed = seed
        self.minNum = minNum
        self.maxNum = maxNum
        self.jitter = jitter
        self.searchFive = searchFive

    def forward_(self, out, points):
        if (self.searchFive):
            return noise_lib.worleyNoise_five_(out, points, self.scale, self.seed, self.minNum, self.maxNum, self.jitter)
        return noise_lib.worleyNoise_(out, points, self.scale, self.seed, self.minNum, self.maxNum, self.jitter)
        

class RepeaterSimplex(NoiseFun):
    def __init__(self, scale, seed, octaves, lacunarity, decay):
        self.scale = scale
        self.seed = seed
        self.octaves = octaves
        self.lacunarity = lacunarity
        self.decay = decay
    
    def forward_(self, out, points):
        return noise_lib.repeaterSimplex_(out, points, self.scale, self.seed, self.octaves, self.lacunarity, self.decay)
        
class RepeaterPerlin(NoiseFun):
    def __init__(self, scale, seed, octaves, lacunarity, decay):
        self.scale = scale
        self.seed = seed
        self.octaves = octaves
        self.lacunarity = lacunarity
        self.decay = decay
    
    def forward_(self, out, points):
        return noise_lib.repeaterPerlin_(out, points, self.scale, self.seed, self.octaves, self.lacunarity, self.decay)
        






class NoiseComposition(NoiseFun):
    def __init__(self, components):
        self.components = components
    
    def advanceSeed(self):
        NoiseFun.advanceSeed(self)
        for c in self.components:
            c.advanceSeed()

class Shifted(NoiseFun):
    def __init__(self, noiseFunc, shift, cuda_device = 0):
        self.noiseFunc = noiseFunc
        self.shift = torch.tensor(shift, dtype=torch.float32, device="cuda:"+str(cuda_device))
    
    def forward_(self, out, points):
        return self.noiseFunc.forward_(out, points + self.shift)

class ScalarAdder(NoiseComposition):
    
    @property
    def amp(self):
        return self.noiseFunc.amp
        
    def __init__(self, noiseFunc, val):
        super().__init__([noiseFunc])
        
        self.noiseFunc = noiseFunc
        self.val = val
    
    
    
    def forward_(self, out, points):
        self.noiseFunc.forward_(out, points)
        out += self.val
        return out

# Multiply the amplitude of the given noise
class Amplifier(NoiseComposition):

    @property
    def amp(self):
        return abs(self.factor) * self.noiseFunc.amp

    def __init__(self, noiseFunc, factor):
        super().__init__([noiseFunc])
        
        self.noiseFunc = noiseFunc
        self.factor = factor
        
    def forward_(self, out, points):
        self.noiseFunc.forward_(out, points)
        out *= self.factor
        return out;

class Adder(NoiseComposition):

    @property
    def amp(self):
        return sum([c.amp for c in self.components])
    
    def __init__(self, components):
        super().__init__(components)
        
    def forward_(self, out, points):
        if len(self.components) == 0:
            out *= 0.0
        else:
            temp_out = torch.empty_like(out)
            out = self.components[0].forward_(out, points)
            for c in self.components[1:]:
                out += c.forward_(temp_out, points)
        return out
            


class BinarySubtractor(NoiseComposition):
    """
    This is a special case of a binary operator, because i assume it is somewhat common.
    """
    @property
    def amp(self):
        return self.first.amp + self.second.amp
    
    def __init__(self, first, second):
        super().__init__([first, second])
        
        self.first = first
        self.second = second
        
    def forward_(self, out, points):
        out = self.first.forward_(out, points)
        temp_out = torch.zeros_like(out)
        temp_out = self.second.forward_(temp_out, points)
        return out.sub_(temp_out)

class BinaryOperator(NoiseComposition):
    
    @property
    def amp(self):
        # This will fail spectacularly
        raise Exception('not implemented')
    
    def __init__(self, first, second, op):
        """
        The binary operator "op" is allowed to modify the first argument, but it doesn't have to.
        It can be implemented like:
            first *= second
            return first
        or like:
            return first * second
        """
        super().__init__([first, second])
        
        self.first = first
        self.second = second
        self.op = op
        
    def forward_(self, out, points):
        out = self.first.forward_(out, points)
        temp_out = torch.zeros_like(out)
        temp_out = self.second.forward_(temp_out, points)
        out = self.op(out, temp_out)
        return out
    

def repeaterFactory(constructinator, seed, octaves, lacunarityFunc = lambda x: x*2.0, initial_params = 1.0, decayFunc = lambda x: x*0.5):
    """
    The constructinator is a function from (seed, params) -> NoiseFunction
    By default, params is a singular floating point value representing scale (default 1)
    and the lacunarity function repeatedly is applied to the params each iteration, in the default case doubling the scale.
    """
    components = []
    amp = 1.0
    params = initial_params
    for i in range(octaves):
        components.append(\
            Amplifier(constructinator(hashInts(seed, i), params), amp)\
        )
    
        params = lacunarityFunc(params)
        amp = decayFunc(amp)
    return Adder(components)
    

def octaveSpots(scale, seed, octaves, lacu_scale=2.0, lacu_spot_size=0.6, decay=1.0, size = 0.5, minNum = 1, maxNum = 3, jitter = 2.0, profileShape = 0):
    """
    This one is a bit more complicated. In addition to increasing the scale, we'll also reduce the size of the spots.
    Furthermore, the amplitude should probably be fixed, so decay is set to 1.0 by default.
    Perhaps this method only reveals how gross the repeaterFactory truly is, who knows.
    """
    constructinator = lambda seed, params: Spots(params[0], seed, params[1], minNum, maxNum, jitter, profileShape)
    decayFunc = lambda x: x*decay
    lacunarityFunc = lambda x: [x[0]*lacu_scale, x[1]*lacu_spot_size]
    
    return repeaterFactory(constructinator, seed, octaves, lacunarityFunc, scale, decayFunc)
    

def octaveWorleyNoise(scale, seed, octaves, lacunarity=2.0, decay=0.5, minNum = 1, maxNum = 1, jitter = 1.0, searchFive = False):
    """
    Hopefully this explains the usage of the repeater factory.
    """
    constructinator = lambda seed, scale: WorleyNoise(scale, seed, minNum, maxNum, jitter, searchFive)
    decayFunc = lambda x: x*decay
    lacunarityFunc = lambda x: x*lacunarity
    
    return repeaterFactory(constructinator, seed, octaves, lacunarityFunc, scale, decayFunc)