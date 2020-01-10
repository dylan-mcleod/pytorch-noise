import torch
import numpy as np
import cv2
import time

from pytorch_noise import *



def benchmark(func):
    
    def anon(*args):
        start = time.time()
        ret = func(*args)
        elapsed = time.time()-start
        return (elapsed, ret)
    
    return anon
    
def FPSCounter():
    totalFrameTime = 0.0
    frames = 0
    def getFPS(next):
        nonlocal totalFrameTime, frames
        frames += 1
        totalFrameTime += next
        if totalFrameTime > 0:
            return frames / totalFrameTime
        else: 
            # at least, probably?
            return 10000
    
    return getFPS
    

def render_noise(noiseFuncs, start, update, step_size, steps):
    """
    Render some 2D noise function over the given range
    """
    
    windowName = "Press 's' to re(S)eed"
    
    
    def innerLoop():
        nonlocal noiseFuncs, start, update, step_size, steps
    
        # Temporary solution, for not having a range2D function :3
        points = range3D(start, step_size, (steps[0], steps[1], 1)).reshape(steps[0], steps[1], 3)
        
        outs = []
        for n in noiseFuncs:
            outs.append((  n(points).atan()  )*0.5+0.5)
            #print(outs[-1].size())
            #print(torch.min(outs[-1]), torch.max(outs[-1]))
            outs.append(torch.zeros(steps[0],1).cuda())
        vals = torch.cat(outs,1)
        #vals = vals[..., None].expand(-1, -1, -1, 3)
        
        cv2.imshow(windowName, vals.detach().cpu().numpy())
        k = cv2.waitKey(1)
        
        if chr(k & 255) == 's':
            for n in noiseFuncs:
                n.advanceSeed()
        
        start += update
    
        if cv2.getWindowProperty(windowName, 0) < 0:
            return False
        else:
            return True
    
    innerLoop = benchmark(innerLoop)
    counter = FPSCounter()
    
    cont = True
    while cont:
        time, cont = innerLoop()
        print("FPS: ", counter(time))
        
    
    