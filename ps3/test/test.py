'''
Created on Jan 23, 2017

@author: Cathy
'''
import unittest
import numpy as np
import edf
from func import Conv, MaxPool, AvePool
###########################################################################################
class TestAvePool(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testForwardShape(self):
        x = type('', (), {})()
        x.value = np.ndarray([5, 10, 15, 30], np.dtype(np.float64))
        x.value.fill(0)
        x.grad = x.value
        
        ave = AvePool(x, 5, 3)
        ave.forward()
        
        self.assertEqual((5, 2, 4, 30), ave.value.shape, "Ave Result Shape ") 
        
    def testForward(self):
        x = type('', (), {})()
        x.value = np.ndarray([2, 4, 6, 2], np.dtype(np.float64))
        x.value.fill(0)
        x.value[1, :, :, 0] = np.transpose([[1, 6, 2, 5], [0, 3, 2, 2], [7 , 9 , 5 , 0], [0, 2, 0, 3], [0, 0, 6, 0], [3, 1, 4, 9]])
        x.grad = x.value
        
        ave = AvePool(x, 3, 2)
        
        ave.forward()
        
        self.assertEqual((2, 1, 2, 2), ave.value.shape)
        self.assertEqual(edf.DT(35) / 9, ave.value[1, 0, 0, 0])
        self.assertEqual(edf.DT(29) / 9, ave.value[1, 0, 1, 0])

        ave2 = AvePool(x, 2, 2)
        ave2.forward()
        self.assertEqual((2, 2, 3, 2), ave2.value.shape)
        self.assertEqual(edf.DT(10) / 4, ave2.value[1, 0, 0, 0])
        self.assertEqual(edf.DT(18) / 4, ave2.value[1, 0, 1, 0])
        self.assertEqual(edf.DT(4) / 4, ave2.value[1, 0, 2, 0])
        self.assertEqual(edf.DT(11) / 4, ave2.value[1, 1, 0, 0])
        self.assertEqual(edf.DT(8) / 4, ave2.value[1, 1, 1, 0])
        self.assertEqual(edf.DT(19) / 4, ave2.value[1, 1, 2, 0])
        

    def testBackward(self):
        x = type('', (), {})()
        x.value = np.ndarray([2, 4, 6, 2], np.dtype(np.float64))
        x.value.fill(0)
        x.grad = np.ndarray([2, 4, 6, 2], np.dtype(np.float64))
        x.grad.fill(0)
        
        ave = AvePool(x, 3, 1)
        ave.value = np.ndarray([2, 2, 4, 2], np.dtype(np.float64))
        ave.value.fill(0)
        ave.grad = np.ndarray([2, 2, 4, 2], np.dtype(np.float64))
        ave.grad.fill(0)
        
        ave.grad[0, :, :, 0] = [[1, 3, 5, 7], [2, 4, 6, 8]]
        
        ave.backward()
        
        
        self.assertEqual(edf.DT(1) / 9, x.grad[0, 0, 0, 0])
        self.assertEqual(edf.DT(4) / 9, x.grad[0, 0, 1, 0])
        self.assertEqual(edf.DT(9) / 9, x.grad[0, 0, 2, 0])
        self.assertEqual(edf.DT(15) / 9, x.grad[0, 0, 3, 0])
        self.assertEqual(edf.DT(12) / 9, x.grad[0, 0, 4, 0])
        self.assertEqual(edf.DT(7) / 9, x.grad[0, 0, 5, 0])
        
        self.assertEqual(edf.DT(3) / 9, x.grad[0, 1, 0, 0])
        self.assertEqual(edf.DT(10) / 9, x.grad[0, 1, 1, 0])
        self.assertEqual(edf.DT(21) / 9, x.grad[0, 1, 2, 0])
        self.assertEqual(edf.DT(33) / 9, x.grad[0, 1, 3, 0])
        self.assertEqual(edf.DT(26) / 9, x.grad[0, 1, 4, 0])
        self.assertEqual(edf.DT(15) / 9, x.grad[0, 1, 5, 0])
        
        self.assertEqual(edf.DT(3) / 9, x.grad[0, 2, 0, 0])
        self.assertEqual(edf.DT(10) / 9, x.grad[0, 2, 1, 0])
        self.assertEqual(edf.DT(21) / 9, x.grad[0, 2, 2, 0])
        self.assertEqual(edf.DT(33) / 9, x.grad[0, 2, 3, 0])
        self.assertEqual(edf.DT(26) / 9, x.grad[0, 2, 4, 0])
        self.assertEqual(edf.DT(15) / 9, x.grad[0, 2, 5, 0])
        
        self.assertEqual(edf.DT(2) / 9, x.grad[0, 3, 0, 0])
        self.assertEqual(edf.DT(6) / 9, x.grad[0, 3, 1, 0])
        self.assertEqual(edf.DT(12) / 9, x.grad[0, 3, 2, 0])
        self.assertEqual(edf.DT(18) / 9, x.grad[0, 3, 3, 0])
        self.assertEqual(edf.DT(14) / 9, x.grad[0, 3, 4, 0])
        self.assertEqual(edf.DT(8) / 9, x.grad[0, 3, 5, 0])

###########################################################################################
class TestMaxPool(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testForwardShape(self): 
        # Test size
        x = type('', (), {})()
        x.value = np.ndarray([5, 10, 15, 30], np.dtype(np.float64))
        x.value.fill(0)
        x.grad = x.value
        
        maxp = MaxPool(x, 5, 3)
        maxp.forward()
        self.assertEqual((5, 2, 4, 30), maxp.value.shape, "Max Result Shape") 

    def testForward(self):
        x = type('', (), {})()
        x.value = np.ndarray([2, 4, 6, 2], np.dtype(np.float64))
        x.value.fill(0)
        x.value[1, :, :, 0] = np.transpose([[1, 6, 2, 5], [0, 3, 2, 2], [7 , 9 , 5 , 0], [0, 2, 0, 3], [0, 0, 6, 0], [3, 1, 4, 9]])
        x.grad = x.value
        
        maxp = MaxPool(x, 2, 2)
        maxp.forward()
        self.assertEqual((2, 2, 3, 2), maxp.value.shape)
        
        self.assertEqual(edf.DT(6), maxp.value[1, 0, 0, 0])
        self.assertEqual(edf.DT(9), maxp.value[1, 0, 1, 0])
        self.assertEqual(edf.DT(3) , maxp.value[1, 0, 2, 0])
        self.assertEqual(edf.DT(5), maxp.value[1, 1, 0, 0])
        self.assertEqual(edf.DT(5) , maxp.value[1, 1, 1, 0])
        self.assertEqual(edf.DT(9), maxp.value[1, 1, 2, 0])

        maxp2 = MaxPool(x, 3, 1)
        maxp2.forward()
        self.assertEqual((2, 2, 4, 2), maxp2.value.shape)

        self.assertEqual(edf.DT(9), maxp2.value[1, 0, 0, 0])
        self.assertEqual(edf.DT(9), maxp2.value[1, 0, 1, 0])
        self.assertEqual(edf.DT(9) , maxp2.value[1, 0, 2, 0])
        self.assertEqual(edf.DT(6) , maxp2.value[1, 0, 3, 0])
        self.assertEqual(edf.DT(9), maxp2.value[1, 1, 0, 0])
        self.assertEqual(edf.DT(9) , maxp2.value[1, 1, 1, 0])
        self.assertEqual(edf.DT(9), maxp2.value[1, 1, 2, 0])
        self.assertEqual(edf.DT(9), maxp2.value[1, 1, 3, 0])

    def testBackward1(self):
        x = type('', (), {})()
        x.value = np.ndarray([2, 4, 6, 2], np.dtype(np.float64))
        x.value.fill(0)
        x.value[1, :, :, 0] = np.transpose([[1, 6, 2, 5], [0, 3, 2, 5], [7 , 9 , 4, 4], [9, 2, 0, 3], [3, 3, 6, 0], [3, 1, 4, 9]])
        
        x.grad = np.ndarray([2, 4, 6, 2], np.dtype(np.float64))
        x.grad.fill(0)
        
        maxp = MaxPool(x, 2, 2)
        maxp.forward()
        
        maxp.grad = np.ndarray([2, 2, 3, 2], np.dtype(np.float64))
        maxp.grad.fill(0)
        maxp.grad[1, :, :, 0] = [[1, 3, 5], [2, 4, 6]]
        maxp.backward()
        
        self.assertEqual(edf.DT(0) , x.grad[1, 0, 0, 0])
        self.assertEqual(edf.DT(0) , x.grad[1, 0, 1, 0])
        self.assertEqual(edf.DT(0) , x.grad[1, 0, 2, 0])
        self.assertEqual(edf.DT(3), x.grad[1, 0, 3, 0])
        self.assertEqual(edf.DT(5), x.grad[1, 0, 4, 0])
        self.assertEqual(edf.DT(5), x.grad[1, 0, 5, 0])
        
        self.assertEqual(edf.DT(1) , x.grad[1, 1, 0, 0])
        self.assertEqual(edf.DT(0) , x.grad[1, 1, 1, 0])
        self.assertEqual(edf.DT(3) , x.grad[1, 1, 2, 0])
        self.assertEqual(edf.DT(0), x.grad[1, 1, 3, 0])
        self.assertEqual(edf.DT(5), x.grad[1, 1, 4, 0])
        self.assertEqual(edf.DT(0), x.grad[1, 1, 5, 0])
        
        self.assertEqual(edf.DT(0) , x.grad[1, 2, 0, 0])
        self.assertEqual(edf.DT(0) , x.grad[1, 2, 1, 0])
        self.assertEqual(edf.DT(4) , x.grad[1, 2, 2, 0])
        self.assertEqual(edf.DT(0), x.grad[1, 2, 3, 0])
        self.assertEqual(edf.DT(0), x.grad[1, 2, 4, 0])
        self.assertEqual(edf.DT(0), x.grad[1, 2, 5, 0])
        
        self.assertEqual(edf.DT(2) , x.grad[1, 3, 0, 0])
        self.assertEqual(edf.DT(2) , x.grad[1, 3, 1, 0])
        self.assertEqual(edf.DT(4) , x.grad[1, 3, 2, 0])
        self.assertEqual(edf.DT(0), x.grad[1, 3, 3, 0])
        self.assertEqual(edf.DT(0), x.grad[1, 3, 4, 0])
        self.assertEqual(edf.DT(6), x.grad[1, 3, 5, 0])
        
    def testBackward2(self):
        x = type('', (), {})()
        x.value = np.ndarray([2, 4, 6, 2], np.dtype(np.float64))
        x.value.fill(0)
        x.value[0, :, :, 0] = np.transpose([[1, 4, 5, 5], [6, 3, 2, 5], [6, 5, 4, 4], [0, 2, 0, 3], [3, 3, 0, 6], [3, 1, 2, 9]])
        
        x.grad = np.ndarray([2, 4, 6, 2], np.dtype(np.float64))
        x.grad.fill(0)
        
        maxp = MaxPool(x, 3, 1)
        maxp.forward()
        self.assertEqual((2, 2, 4, 2), maxp.value.shape, '')
        maxp.grad = np.ndarray([2, 2, 4, 2], np.dtype(np.float64))
        maxp.grad.fill(0)
        maxp.grad[0, :, :, 0] = [[1, 2, 3, 4], [5, 6, 7, 8]]
        maxp.backward()    

        self.assertEqual(edf.DT(0) , x.grad[0, 0, 0, 0])
        self.assertEqual(edf.DT(3) , x.grad[0, 0, 1, 0])
        self.assertEqual(edf.DT(6) , x.grad[0, 0, 2, 0])
        self.assertEqual(edf.DT(0), x.grad[0, 0, 3, 0])
        self.assertEqual(edf.DT(4), x.grad[0, 0, 4, 0])
        self.assertEqual(edf.DT(4), x.grad[0, 0, 5, 0])
        
        self.assertEqual(edf.DT(0) , x.grad[0, 1, 0, 0])
        self.assertEqual(edf.DT(0) , x.grad[0, 1, 1, 0])
        self.assertEqual(edf.DT(11) , x.grad[0, 1, 2, 0])
        self.assertEqual(edf.DT(0), x.grad[0, 1, 3, 0])
        self.assertEqual(edf.DT(4), x.grad[0, 1, 4, 0])
        self.assertEqual(edf.DT(0), x.grad[0, 1, 5, 0])
        
        self.assertEqual(edf.DT(5) , x.grad[0, 2, 0, 0])
        self.assertEqual(edf.DT(0) , x.grad[0, 2, 1, 0])
        self.assertEqual(edf.DT(0) , x.grad[0, 2, 2, 0])
        self.assertEqual(edf.DT(0), x.grad[0, 2, 3, 0])
        self.assertEqual(edf.DT(0), x.grad[0, 2, 4, 0])
        self.assertEqual(edf.DT(0), x.grad[0, 2, 5, 0])
        
        self.assertEqual(edf.DT(5) , x.grad[0, 3, 0, 0])
        self.assertEqual(edf.DT(11) , x.grad[0, 3, 1, 0])
        self.assertEqual(edf.DT(0) , x.grad[0, 3, 2, 0])
        self.assertEqual(edf.DT(0), x.grad[0, 3, 3, 0])
        self.assertEqual(edf.DT(7), x.grad[0, 3, 4, 0])
        self.assertEqual(edf.DT(8), x.grad[0, 3, 5, 0])

###########################################################################################
class TestConv(unittest.TestCase):
    
    def setUp(self):
        pass

    def tearDown(self):
        pass


    def testForward(self):
        self.fail("Not implemented")

    def testBackward(self):
        self.fail("Not implemented")
    

if __name__ == '__main__':
    unittest.main()    
