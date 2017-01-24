'''
Created on Jan 23, 2017

@author: Cathy
'''
import numpy as np
import edf
from math import ceil


class Conv:
    def __init__(self, f, k, stride=1, pad=0):
        edf.components.append(self)
        self.f = f
        self.k = k
        pad = np.array(pad)
        if pad.shape == ():
            self.xpad = self.ypad = pad
        else:
            self.ypad = pad[0]
            self.xpad = pad[1]
            
        self.stride = stride
        self.grad = None if f.grad is None and k.grad is None else edf.DT(0) 

    ####################### Please implement this function####################### 
    def forward(self):
        inshape = self.k.shape
        if self.value is None:
            owsize = np.ceil((inshape[1] - self.f.shape[0] + 1) / self.stride)
            ohsize = np.ceil((inshape[2] - self.f.shape[0] + 1) / self.stride)
            self.value = np.ndarray([inshape[0], owsize, ohsize, inshape[3]],
                                    np.dtype(np.float64))
        
        # padding
        padded = np.ndarray([inshape[0], inshape[1] + 2 * self.pad, inshape[2] + 2 * self.pad, inshape[3]])
        padded.fill(0)
        padded[:, self.pad: inshape[1] + self.pad , self.pad : inshape[2] + self.pad , :] = self.k.value
        yshape = self.value.shape
        ksize = self.f.shape[0]
        for bi in range(yshape[0]):
            for ci in range(yshape[3]):
                for wi in range(yshape[1]):
                    for hi in range(yshape[2]):
                        self.value[bi, wi, hi, ci] = np.multiply(padded[bi,
                                                                        self.stride * wi : self.stride * wi + ksize,
                                                                        self.stride * hi : self.stride * hi + ksize,
                                                                        :],
                                                                 self.f[:, :, :, ci]).sum()
                
    ####################### Please implement this function#######################         
    def backward(self):
        if self.k.grad is None or self.f.grad is None:
            return
        fgrad = np.ndarray(self.f.value.shape, np.dtype(np.float64))
        fgrad.fill(0)
        kgrad = np.ndarray(self.k.value.shape, np.dtype(np.float64))
        kgrad.fill(0)
        
        kshape = self.k.value.shape
        fshape = self.f.value.shape
        yshape = self.value.shape
        for bi in range(kshape[0]):
            for ci in range(kshape[3]):
                for c2i in range(kshape[3]):
                    for wi in range(yshape[1]):
                        for hi in range(yshape[2]):
                            for wki in range(fshape[0]):
                                for hki in range(fshape[0]):
                                    xwi = self.stride * wi + wki
                                    xhi = self.stride * hi + hki
                                    kgrad[bi, xwi, xhi, ci] += self.grad[bi, wi, hi, c2i] * self.f.value[wki, hki, c2i, ci]
                                    fgrad[wki, hki, c2i, ci] += self.grad[bi, wi, hi, c2i] * self.k.value[bi, xwi, xhi, ci]
        self.f.grad += fgrad
        self.k.grad += kgrad

########################################### MaxPool layer#############################################
############################### Please implement the forward abd backward method in this class ##############             
class MaxPool:
    def __init__(self, x, ksz=2, stride=None):
        edf.components.append(self)
        self.x = x
        self.ksz = ksz
        if stride is None:
            self.stride = ksz
        else:
            self.stride = stride
        self.grad = None if x.grad is None else edf.DT(0)

    ####################### Please implement this function#######################     
    def forward(self):
        xshape = self.x.value.shape
        self.value = np.ndarray([xshape[0],
                                 int(ceil((xshape[1] - self.ksz + 1) / self.stride)),
                                 int(ceil((xshape[2] - self.ksz + 1) / self.stride)),
                                 xshape[3]], self.x.value.dtype)
        self.xmaxs = {}
        vshape = self.value.shape
        for bi in range(vshape[0]):
            for ci in range(vshape[3]):
                for wi in range(vshape[1]):
                    for hi in range(vshape[2]):
                        xmax = np.NINF
                        for swi in range(self.ksz):
                            for shi in range(self.ksz):
                                # Record the maximal value and its location
                                xvalue = self.x.value[bi, wi * self.stride + swi,
                                                      hi * self.stride + shi, ci]
                                record = (bi, wi * self.stride + swi, hi * self.stride + shi, ci)
                                if xvalue > xmax:
                                    xmax = xvalue
                                    self.xmaxs[(bi, wi, hi, ci)] = [record]
                                elif xvalue == xmax:
                                    self.xmaxs[(bi, wi, hi, ci)].append(record)
                        self.value[bi, wi, hi, ci] = xmax
                        
    ####################### Please implement this function#######################             
    def backward(self):
        if self.x.grad is None:
            return
        grad = np.ndarray(self.x.value.shape, np.dtype(np.float64))
        grad.fill(0)
        # for each grad, prop only to the max inputs
        sshape = self.value.shape
        for bi in range(sshape[0]):
            for ci in range(sshape[3]):
                for wi in range(sshape[1]):
                    for hi in range(sshape[2]):
                        gval = self.grad[bi, wi, hi, ci]
                        for xmaxp in self.xmaxs[(bi, wi, hi, ci)]:
                            grad[xmaxp[0], xmaxp[1], xmaxp[2], xmaxp[3]] += gval
        self.x.grad += grad
########################################### AvePool layer#############################################
############################### Please implement the forward abd backward method in this class ##############                             
class AvePool:
    def __init__(self, x, ksz=2, stride=None):
        edf.components.append(self)
        self.x = x
        self.ksz = ksz
        if stride is None:
            self.stride = ksz
        else:
            self.stride = stride
        self.grad = None if x.grad is None else edf.DT(0)
        
    ####################### Please implement this function#######################   
    def forward(self):
        xshape = self.x.value.shape
        self.value = np.ndarray([xshape[0],
                                 int(ceil((xshape[1] - self.ksz + 1) / self.stride)),
                                 int(ceil((xshape[2] - self.ksz + 1) / self.stride)),
                                 xshape[3]], self.x.value.dtype)
        vshape = self.value.shape
        for bi in range(vshape[0]):
            for ci in range(vshape[3]):
                for wi in range(vshape[1]):
                    for hi in range(vshape[2]):
                        xsum = edf.DT(0)
                        for swi in range(self.ksz):
                            for shi in range(self.ksz):
                                xsum += self.x.value[bi, wi * self.stride + swi,
                                                      hi * self.stride + shi, ci]
                        xavg = xsum / (np.square(self.ksz))
                        self.value[bi, wi, hi, ci] = xavg
    ####################### Please implement this function#######################    
    def backward(self):
        if self.x.grad is None:
            return
        grad = np.ndarray(self.x.value.shape, np.dtype(np.float64))
        grad.fill(0)
        vshape = self.value.shape
        for bi in range(vshape[0]):
            for ci in range(vshape[3]):
                for wi in range(vshape[1]):
                    for hi in range(vshape[2]):
                        gval = self.grad[bi, wi, hi, ci]
                        for wsi in range(self.ksz):
                            for hsi in range(self.ksz):
                                grad[bi, wi * self.stride + wsi, hi * self.stride + hsi, ci] += gval
        
        self.x.grad += grad / np.square(self.ksz)


