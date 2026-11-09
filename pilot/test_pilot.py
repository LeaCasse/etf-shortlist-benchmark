import unittest
import numpy as np

from .model import BITS, GROUPS, evaluate, instance, proxy, short_mask
from .samplers import state

class PilotChecks(unittest.TestCase):
    def test_convex_certificate_and_equal_weight(self):
        p=instance('aligned',201)
        ids=np.where(short_mask(p,4))[0]
        x=[evaluate(p,int(j)) for j in ids]
        self.assertTrue(any(v['equal_ok'] for v in x))
        for v in x:
            if v['equal_ok']:
                self.assertLessEqual(v['cost'],v['equal_cost']+1e-8)
                self.assertLessEqual(v['certificate_gap'],2e-6)
                self.assertLessEqual(v['violation'],1e-8)

    def test_infeasibility_not_a_numerical_failure(self):
        p=instance('tight',301)
        p.yield_floor=.99
        v=evaluate(p,0)
        self.assertEqual(v['status'],'infeasible')
        self.assertIsNone(v['cost'])

    def test_xy_matches_full_register(self):
        e=proxy(instance('complement',101))
        e=(e-e.mean())/e.std()
        gamma,beta=.8,-.4
        enc=BITS@(1<<np.arange(12))
        full=np.zeros(2**12,dtype=complex)
        full[enc]=np.exp(-1j*gamma*e)/np.sqrt(len(enc))
        all_states=np.arange(len(full))
        for g in GROUPS:
            for i,j in zip(g,g[1:]+g[:1]):
                active=all_states[((all_states>>i)&1)!=((all_states>>j)&1)]
                full[active]=np.cos(beta)*full[active]-1j*np.sin(beta)*full[active^((1<<i)|(1<<j))]
        outside=np.ones(len(full),bool);outside[enc]=False
        self.assertLess(np.linalg.norm(full[outside]),1e-12)
        np.testing.assert_allclose(full[enc],state(e,gamma,beta),atol=1e-12,rtol=0)
        self.assertAlmostEqual(np.linalg.norm(full),1.,places=12)


if __name__ == '__main__':
    unittest.main()
