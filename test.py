import random
import unittest
import pyBinMixt
import numpy as np

class Test(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.x = np.concatenate((np.random.multivariate_normal([-4,-5],np.diag([1,.5]) , 7000), np.random.multivariate_normal([4,5],np.diag([1,.5]) , 3000)), axis=0)
        self.y = np.concatenate([np.repeat(0,7000),np.repeat(1,3000)])
        self.grid=[20,20]
        self.classes=2
        self.it_init=10
        self.it_algo=100
        self.eps_init=0.00001
        self.eps_algo=0.00001
        self.n_init=1
        self.seed=1
        self.b = pyBinMixt.BinMixt(self.classes, self.grid, self.it_init, self.it_algo, self.eps_init, self.eps_algo, self.n_init,self.seed)
        self.b.fit(self.x)
    def test_input(self):
        self.assertEqual(self.b.grid,[20,20])
        self.assertEqual(self.b.classes, 2)
        self.assertEqual(self.b.it_algo, 100)
        self.assertEqual(self.b.it_init,10)
        self.assertEqual(self.b.eps_init,0.00001)
        self.assertEqual(self.b.eps_algo,0.00001)
        self.assertEqual(self.b.n_init,1)
        self.assertEqual(self.b.seed, 1)
        self.assertEqual(self.x[0,0],-2.3756546363367583)
        self.assertEqual(self.x[0,1],-5.4325771085263312)
    def test_fit(self):
        self.assertEqual(self.b.pi_,[0.6213220646609793, 0.37867793533902067])
        self.assertEqual(self.b.mu_,[[-0.22737908398165307, -3.216588327411712], [-5.0158884025513455, 4.860997680346315]])
        self.assertEqual(self.b.v_,[[16.98598328535709, 6.66411760143277], [0.47905557236400004, 1.6435309857823768]])
    def test_fit_predict(self):
        pred=self.b.fit_predict(self.x)
        self.assertEqual(pred.tolist()[6995:7005],[0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    def test_score(self):
        self.assertEqual(self.b.score(self.x,self.y),1)
    def test_window(self):
        x1=self.x[0:10,:]
        wt=pyBinMixt.WindowTransformer(8)
        self.assertEqual(wt.fit_transform(x1,self.y[0:10])[0],[0.8766973574019544, -3.2126384654543814, 3.315655196810007, 0.5309841124731095, -5.754918455014602, 5.776312624469364, 0.0])
        wt = pyBinMixt.WindowTransformer(8,False)
        self.assertEqual(wt.fit_transform(x1,self.y[0:10])[0],[-3.2126384654543814, 3.315655196810007,  -5.754918455014602, 5.776312624469364, 0.0])
        wt = pyBinMixt.WindowTransformer(8, False,False)
        self.assertEqual(wt.fit_transform(x1, self.y[0:10])[0],
                         [3.315655196810007,5.776312624469364, 0.0])


if __name__ == "__main__":
    unittest.main()