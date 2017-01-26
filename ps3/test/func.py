'''
Created on Jan 23, 2017

@author: Cathy
'''
import numpy as np
import edf


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
        xshape = self.k.value.shape
        fshape = self.f.value.shape
        
        # padding
        padded = np.ndarray([xshape[0],
                             xshape[1] + 2 * self.xpad,
                             xshape[2] + 2 * self.ypad,
                             xshape[3]],
                            self.k.value.dtype)
        padded.fill(0)
        padded[:, self.xpad: xshape[1] + self.xpad , self.ypad : xshape[2] + self.ypad , :] = self.k.value
        pshape = padded.shape
        # Calculate shape
        owsize = int(np.ceil((pshape[1] - fshape[0] + 1) / self.stride))
        ohsize = int(np.ceil((pshape[2] - fshape[1] + 1) / self.stride))
        self.value = np.ndarray([xshape[0], owsize, ohsize, fshape[3]], np.dtype(np.float64))
        yshape = self.value.shape
        
        for wi in range(yshape[1]):
            for hi in range(yshape[2]):
                slide = padded[:, self.stride * wi : self.stride * wi + fshape[0],
                               self.stride * hi : self.stride * hi + fshape[1], :]
                self.value[:, wi, hi, :] = np.einsum('buvc,uvcp->bp', slide, self.f.value)
                
    ####################### Please implement this function#######################         
    def backward(self):
        if self.k.grad is None or self.f.grad is None:
            return
        
        kshape = self.k.value.shape
        fshape = self.f.value.shape
        fgrad = np.ndarray(fshape, np.dtype(np.float64))
        fgrad.fill(0)
        pad_kgrad = np.ndarray([kshape[0], kshape[1] + 2 * self.xpad,
                                kshape[2] + 2 * self.ypad, kshape[3]],
                               np.dtype(np.float64))
        pad_kgrad.fill(0)
        pad_kval = np.ndarray(pad_kgrad.shape, self.k.value.dtype)
        pad_kval.fill(0)
        pad_kval[:, self.xpad: kshape[1] + self.xpad , self.ypad : kshape[2] + self.ypad , :] = self.k.value
        
        
        yshape = self.value.shape
        
        for wi in range(yshape[1]):
            for hi in range(yshape[2]):
                woffset = self.stride * wi
                hoffset = self.stride * hi
                grad_slide = self.grad[:, wi, hi, :]
                pad_kgrad[:, woffset: woffset + fshape[0], hoffset:hoffset + fshape[1], :] += np.einsum('ij,kmnj->ikmn', grad_slide, self.f.value)
                
                xslide = pad_kval[:, woffset: woffset + fshape[0], hoffset:hoffset + fshape[1], :]
            
                fgrad += np.einsum('imnk,ij->mnkj', xslide, grad_slide)
                                    
        self.f.grad += fgrad
        self.k.grad += pad_kgrad[:, self.xpad:self.xpad + kshape[1], self.ypad:self.ypad + kshape[2], :]

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
        
        self.square = np.ndarray([self.ksz, self.ksz])
        self.square.fill(1)

    ####################### Please implement this function#######################     
    def forward(self):
        xshape = self.x.value.shape
        self.value = np.ndarray([xshape[0],
                                 int(np.ceil((xshape[1] - self.ksz + 1) / self.stride)),
                                 int(np.ceil((xshape[2] - self.ksz + 1) / self.stride)),
                                 xshape[3]], self.x.value.dtype)
        self.pattern = np.ndarray(xshape, self.x.value.dtype)
        self.pattern.fill(0)
        vshape = self.value.shape
        for wi in range(vshape[1]):
            for hi in range(vshape[2]):
                # Record the maximal value and its location
                sliding = self.x.value[:, wi * self.stride:wi * self.stride + self.ksz,
                                       hi * self.stride:hi * self.stride + self.ksz,
                                       :]
                maxval = sliding.max(axis=(1, 2))
                self.value[:, wi, hi, :] = maxval
    ####################### Please implement this function#######################             
    def backward(self):
        if self.x.grad is None:
            return
        grad = np.ndarray(self.x.value.shape, np.dtype(np.float64))
        grad.fill(0)
        vshape = self.value.shape
        
        for wi in range(vshape[1]):
            for hi in range(vshape[2]):
                gval = self.grad[:, wi, hi, :]
                sval = self.value[:, wi, hi, :]
                target = self.x.value[:, wi * self.stride:wi * self.stride + self.ksz,
                                      hi * self.stride : hi * self.stride + self.ksz, :]
                gexpand = np.einsum('ij,mn->imnj', gval, self.square)
                vexpand = np.einsum('ij,mn->imnj', sval, self.square)
                equ = np.equal(vexpand, target)
                grad[:, wi * self.stride:wi * self.stride + self.ksz,
                     hi * self.stride : hi * self.stride + self.ksz, :] += gexpand * equ
        
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
        
        self.square = np.ndarray([self.ksz, self.ksz])
        self.square.fill(1)
        
    ####################### Please implement this function#######################   
    def forward(self):
        xshape = self.x.value.shape
        self.value = np.ndarray([xshape[0],
                                 int(np.ceil((xshape[1] - self.ksz + 1) / self.stride)),
                                 int(np.ceil((xshape[2] - self.ksz + 1) / self.stride)),
                                 xshape[3]], self.x.value.dtype)
        vshape = self.value.shape
        for wi in range(vshape[1]):
            for hi in range(vshape[2]):
                xavg = np.average(self.x.value[:, wi * self.stride: wi * self.stride + self.ksz,
                                    hi * self.stride: hi * self.stride + self.ksz, :],
                                  axis=(1, 2))
                self.value[:, wi, hi, :] = xavg
    ####################### Please implement this function#######################    
    def backward(self):
        if self.x.grad is None:
            return
        grad = np.ndarray(self.x.value.shape, np.dtype(np.float64))
        grad.fill(0)
        vshape = self.value.shape
        
        for wi in range(vshape[1]):
            for hi in range(vshape[2]):
                gval = self.grad[:, wi, hi, :]
                gvalexpand = np.einsum('ij,mn->imnj', gval, self.square)
                grad[:, wi * self.stride: wi * self.stride + self.ksz,
                     hi * self.stride: hi * self.stride + self.ksz, :] += gvalexpand 
        self.x.grad += grad / np.square(self.ksz)


