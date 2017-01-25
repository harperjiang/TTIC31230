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

    def testPadding(self):
        x = type('', (), {})()
        x.value = np.ndarray([1, 10, 10, 1])
        x.grad = x.value
        f = type('', (), {})()
        f.value = np.ndarray([1, 1, 1, 1])
        f.grad = f.value
        
        # xpad = ypad = 5
        conv = Conv(f, x, 1, 5)
        conv.forward()
        
        self.assertEqual((1, 20, 20, 1), conv.value.shape)
        
        # xpad = 6, ypad = 8
        conv2 = Conv(f, x, 1, [8, 6])
        conv2.forward()
        
        self.assertEqual((1, 22, 26, 1), conv2.value.shape)
        
    def testForwardShape(self):
        x = type('', (), {})()
        x.value = np.ndarray([3, 10, 10, 4])
        x.grad = x.value
        f = type('', (), {})()
        f.value = np.ndarray([2, 3, 4, 6])
        f.grad = f.value
        
        conv = Conv(f, x, 2, 0)
        conv.forward()
        self.assertEqual((3, 5, 4, 6), conv.value.shape)

    def testForward1(self):
        x = type('', (), {})()
        x.value = np.ndarray([3, 5, 6, 2])
        x.value.fill(0)
        x.value[0, :, :, 0] = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30]]
        x.value[0, :, :, 1] = [[1, 2, 3, 3, 2, 1], [1, 1, 2, 2, 1, 1], [1, 1, 1, 1, 1, 1], [1, 2, 3, 3, 2, 1], [1, 1, 2, 2, 1, 1]]
        
        x.grad = x.value
        f = type('', (), {})()
        f.value = np.ndarray([2, 3, 2, 1])
        f.value[:, :, 0, 0] = [[-1, 2, -1], [-1, 2, -1]]
        f.value[:, :, 1, 0] = [[1, 0, -1], [1, 0, -1]]
        f.grad = f.value
        
        conv = Conv(f, x, 2, 0)
        conv.forward()
        self.assertEqual((3, 2, 2, 1), conv.value.shape)
        
        self.assertEqual(edf.DT(-3), conv.value[0, 0, 0, 0])
        self.assertEqual(edf.DT(2), conv.value[0, 0, 1, 0])
        self.assertEqual(edf.DT(-2), conv.value[0, 1, 0, 0])
        self.assertEqual(edf.DT(1), conv.value[0, 1, 1, 0])
        
    def testForward2(self):    
        x = type('', (), {})()
        x.value = np.ndarray([3, 5, 6, 1])
        x.value.fill(0)
        x.value[0, :, :, 0] = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30]]
        x.grad = x.value
        
        f = type('', (), {})()
        f.value = np.ndarray([2, 3, 1, 2])
        f.value[:, :, 0, 0] = [[1, 2, 1], [2, 2, 1]]
        f.value[:, :, 0, 1] = [[1, 0, -1], [1, 0, -1]]
        f.grad = f.value
        
        conv = Conv(f, x, 3, 0)
        conv.forward()
        self.assertEqual((3, 2, 2, 2), conv.value.shape)
        
        self.assertEqual(edf.DT(47), conv.value[0, 0, 0, 0])
        self.assertEqual(edf.DT(74), conv.value[0, 0, 1, 0])
        self.assertEqual(edf.DT(209), conv.value[0, 1, 0, 0])
        self.assertEqual(edf.DT(236), conv.value[0, 1, 1, 0])
        
        self.assertEqual(edf.DT(-4), conv.value[0, 0, 0, 1])
        self.assertEqual(edf.DT(-4), conv.value[0, 0, 1, 1])
        self.assertEqual(edf.DT(-4), conv.value[0, 1, 0, 1])
        self.assertEqual(edf.DT(-4), conv.value[0, 1, 1, 1])

    def testPadForward(self):
        x = type('', (), {})()
        x.value = np.ndarray([3, 5, 6, 2])
        x.value.fill(0)
        x.value[0, :, :, 0] = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30]]
        x.value[0, :, :, 1] = [[1, 2, 3, 3, 2, 1], [1, 1, 2, 2, 1, 1], [1, 1, 1, 1, 1, 1], [1, 2, 3, 3, 2, 1], [1, 1, 2, 2, 1, 1]]
        
        x.grad = x.value
        f = type('', (), {})()
        f.value = np.ndarray([2, 3, 2, 1])
        f.value[:, :, 0, 0] = [[1, 1, 1], [1, 1, 1]]
        f.value[:, :, 1, 0] = [[1, 0, -1], [1, 0, -1]]
        f.grad = f.value
        
        conv = Conv(f, x, 3, 1)
        conv.forward()
        self.assertEqual((3, 2, 2, 1), conv.value.shape)
        
        self.assertEqual(edf.DT(1), conv.value[0, 0, 0, 0])
        self.assertEqual(edf.DT(13), conv.value[0, 0, 1, 0])
        self.assertEqual(edf.DT(63), conv.value[0, 1, 0, 0])
        self.assertEqual(edf.DT(115), conv.value[0, 1, 1, 0])
        
    def testBackward1(self):
        x = type('', (), {})()
        x.value = np.ndarray([1, 5, 6, 1])
        x.value.fill(0)
        x.value[0, :, :, 0] = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30]]
        
        x.grad = np.ndarray(x.value.shape)
        x.grad.fill(0)
        
        f = type('', (), {})()
        f.value = np.ndarray([2, 3, 1, 1])
        f.value[:, :, 0, 0] = [[1, 1, 1], [1, 2, 1]]
        f.grad = np.ndarray(f.value.shape)
        f.grad.fill(0)
        
        conv = Conv(f, x, 2, 0)
        conv.forward()
        
        conv.grad = np.ndarray([1, 2, 2, 1], np.dtype(np.float64))
        conv.grad[0, :, :, 0] = [[3, 2 ], [7, 5]]
        conv.backward()
        
        self.assertTrue(np.array_equal([[175, 192, 209], [277, 294, 311]], f.grad[:, :, 0, 0]))
        self.assertTrue(np.array_equal([[3, 3, 5, 2, 2, 0], [3, 6, 5, 4, 2, 0], [7, 7, 12, 5, 5, 0], [7, 14, 12, 10, 5, 0], [0, 0, 0, 0, 0, 0]], x.grad[0, :, :, 0]))
    
    def testBackward2(self):
        x = type('', (), {})()
        x.value = np.ndarray([1, 4, 5, 2])
        x.value.fill(0)
        x.value[0, :, :, 0] = [[1, 4, 9, 2, 5], [3, 8, 6, 4, 7], [2, 5, 2, 1, 4], [3, 3, 0, 9, 7]]
        x.value[0, :, :, 1] = [[1, 1, 2, 1, 1], [3, 2, 1, 4, 1], [5, 1, 2, 6, 4], [7, 7, 0, 3, 3]]
        
        x.grad = np.ndarray(x.value.shape)
        x.grad.fill(0)
        
        f = type('', (), {})()
        f.value = np.ndarray([2, 3, 2, 1])
        f.value[:, :, 0, 0] = [[1, 0, 1], [-1, 0, 1]]
        f.value[:, :, 1, 0] = [[1, 2, 1], [1, 2, 1]]
        f.grad = np.ndarray(f.value.shape)
        f.grad.fill(0)
        
        conv = Conv(f, x, 3, 1)
        conv.forward()
                
        conv.grad = np.ndarray([1, 2, 2, 1], np.dtype(np.float64))
        conv.grad[0, :, :, 0] = [[3, 2 ], [7, 5]]
        conv.backward()
        
        self.assertTrue(np.array_equal([[10, 19, 55], [18, 73, 78]], f.grad[:, :, 0, 0]))
        self.assertTrue(np.array_equal([[10, 65, 27], [4, 69, 69]], f.grad[:, :, 1, 0]))
        self.assertTrue(np.array_equal([[0, 3, -2, 0, 2], [0, 0, 0, 0, 0], [0, 7, 5, 0, 5], [0, 7, -5, 0, 5]], x.grad[0, :, :, 0]))
        self.assertTrue(np.array_equal([[6, 3, 2, 4, 2], [0, 0, 0, 0, 0], [14, 7, 5, 10, 5], [14, 7, 5, 10, 5]], x.grad[0, :, :, 1]))
    

if __name__ == '__main__':
    unittest.main()    
