import unittest
import numpy as np
from grain_benchmark.evaluation import clustering_metrics, structured_confusion, macro_silhouette, align_predictions

class EvaluationTests(unittest.TestCase):
    def test_perfect_permuted_noncontiguous(self):
        r=clustering_metrics(np.array([0,0,1,1]),np.array([99,99,7,7]))
        for k in ('ACC','NMI','ARI'): self.assertAlmostEqual(r[k],1)
        self.assertEqual(r['K_occupied'],2)
    def test_hungarian_not_many_to_one(self):
        r=clustering_metrics(np.array([0,0,0,0,1,1]),np.array([0,0,1,1,2,2]))
        self.assertAlmostEqual(r['ACC'],4/6)
    def test_merge(self):
        r=clustering_metrics(np.array([0,0,1,1]),np.array([3,3,3,3]))
        self.assertEqual(r['ACC'],.5)
    def test_strings(self):
        r=clustering_metrics(np.array(['A','B','A']),np.array(['z','x','z']))
        self.assertEqual(r['ACC'],1)
    def test_negative_unassigned_rejected(self):
        with self.assertRaises(ValueError):clustering_metrics(np.array([0,1]),np.array([0,-1]))
    def test_empty(self):
        with self.assertRaises(ValueError):clustering_metrics(np.array([],dtype=int),np.array([],dtype=int))
    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):clustering_metrics(np.array([0,1]),np.array([0]))
    def test_structured_perfect(self):
        s,_=structured_confusion(np.array([0,0,1,1]),np.array([9,9,7,7]))
        np.testing.assert_allclose(s,np.array([[100,0],[0,100],[0,0],[0,0]]))
    def test_structured_fragmentation(self):
        s,_=structured_confusion(np.array([0,0,0,0,1,1]),np.array([0,0,1,1,2,2]))
        np.testing.assert_allclose(s,np.array([[50,0],[0,100],[50,0],[0,0]]))
        np.testing.assert_allclose(s.sum(axis=0),100)
    def test_structured_merging(self):
        s,_=structured_confusion(np.array([0,0,0,1,1]),np.zeros(5,dtype=int))
        np.testing.assert_allclose(s,np.array([[100,100],[0,0],[0,0],[0,0]]))
    def test_structured_invariant_to_cluster_names(self):
        y=np.array([0,0,1,0,0,1,1,1,2,2]);z=np.array([5,5,5,2,2,2,8,8,9,9])
        a,_=structured_confusion(y,z);b,_=structured_confusion(y,100-z)
        np.testing.assert_allclose(a,b)
    def test_align_ids(self):
        p=align_predictions(np.array([0,1,2]),np.array([2,0,1]),np.array([8,9,7]))
        np.testing.assert_array_equal(p,[9,7,8])
    def test_incomplete_ids(self):
        with self.assertRaises(ValueError):align_predictions(np.array([0,1,2]),np.array([0,1]),np.array([5,6]))
    def test_duplicate_ids(self):
        with self.assertRaises(ValueError):align_predictions(np.array([0,1]),np.array([0,0]),np.array([5,6]))
    def test_macro_silhouette_equal_weight(self):
        x=np.array([[1.,0.],[1.,.1],[.9,.2],[0.,1.],[.1,1.]])
        y=np.array([0,0,0,1,1]);r=macro_silhouette(x,y)
        self.assertAlmostEqual(r['macro_silhouette'],np.mean(list(r['per_class'].values())))
    def test_singleton_zero(self):
        r=macro_silhouette(np.array([[1.,0.],[.9,.1],[0.,1.]]),np.array([0,0,1]))
        self.assertEqual(r['per_class']['1'],0)
    def test_silhouette_zero_vector(self):
        with self.assertRaises(ValueError):macro_silhouette(np.array([[0.,0.],[1.,0.],[0.,1.]]),np.array([0,0,1]))

if __name__=='__main__':unittest.main()
