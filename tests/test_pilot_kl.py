import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('pilot_kl',Path(__file__).resolve().parents[1]/'builds/profile_pilot_kl.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class StreamingKLTests(unittest.TestCase):
    def test_independent_probability_oracle_and_chunk_edges(self):
        p=np.array([[.1,.2,.7],[.5,.25,.25],[.001,.009,.99]])
        q=np.array([[.2,.2,.6],[.1,.3,.6],[.01,.09,.9]])
        expected=np.sum(p*np.log(p/q),axis=1)
        for chunk in (1,2,8):
            result=m.chunked_kl(np.log(p),np.log(q),chunk)
            np.testing.assert_allclose(result['token_kl'],expected,rtol=1e-12,atol=1e-12)
            self.assertEqual(result['tokens'],3)
            self.assertEqual(m.chunked_kl(np.log(p),np.log(p),chunk)['mean_kl_nats'],0.)

    def test_logit_shift_and_nonfinite_rejection(self):
        a=np.array([[1000.,999.,-1000.]])
        self.assertAlmostEqual(m.chunked_kl(a,a+5000)['mean_kl_nats'],0.)
        with self.assertRaises(ValueError):
            m.chunked_kl(a,np.array([[np.nan,0,1]]))
        with self.assertRaises(ValueError):
            m.chunked_kl(a,a[:,:2])


if __name__=='__main__':
    unittest.main()
