"""Run at checkout root: python -m unittest discover -s tests -p test_engineering_dropin.py -v"""
from __future__ import annotations
import json, hashlib, math, unittest, tempfile
from pathlib import Path
import numpy as np
from src.engineering import EngineeringVolume, evaluate_volume, compare_pair, from_companion


def fixture():
    dims=(16,10,8)
    alpha=np.ones(dims,dtype=float)*.85
    y=np.zeros(dims,dtype=np.int16)
    y[:4,:,0:3]=1; y[4:7,:,0:3]=2; y[7:11,:,0:3]=5
    y[11:,:,0:3]=7; y[:5,:,3:]=4; y[5:,:,3:]=3
    return EngineeringVolume(y,alpha,'OSV',80.,18.,8.,8,sample_id='fixture')


class EngineeringTests(unittest.TestCase):
    def test_source_integrity(self):
        root=Path(__file__).resolve().parents[1]/'src'/'engineering'
        manifest=json.loads((root/'UPSTREAM_SHA256.json').read_text())
        source={**manifest['original_p1a_sources'], **manifest.get('packaged_text_only_updates',{})}
        for name,h in source.items():
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),h,name)

    def test_identical_pair_zero_delta(self):
        v=fixture();r=compare_pair(v,v)
        self.assertEqual(r['delta_LCG_over_L'],0.)
        self.assertEqual(r['delta_KG_over_D'],0.)
        self.assertEqual(r['delta_J_adj'],0.)
        self.assertTrue(r['pair_adjacency_valid'])

    def test_weights_and_jadj_finite(self):
        r=evaluate_volume(fixture())
        self.assertTrue(r['weight_valid'])
        self.assertTrue(r['adjacency_valid'])
        self.assertTrue(np.isfinite([r['LCG_m'],r['KG_m'],r['J_adj']]).all())
        self.assertAlmostEqual(r['total_mass_t'],r['reference_displacement_t'],6)
        self.assertTrue(np.isnan(r['GM_corrected_m']))

    def test_semantic_mutation_affects_metrics(self):
        v=fixture();labs=v.labels.copy();labs[:4,:,0:3]=5
        new=EngineeringVolume(labs,v.hull_fraction,v.family,v.L,v.B,v.D,v.nz_hull)
        comp=compare_pair(v,new)
        self.assertNotAlmostEqual(comp['delta_LCG_over_L'],0)
        self.assertFalse(comp['prediction']['adjacency_valid'])
        self.assertTrue(math.isnan(comp['delta_J_adj']))

    def test_equilibrium_optional(self):
        r=evaluate_volume(fixture(),equilibrium=True)
        self.assertEqual(r['trim_status'],'ok')
        self.assertTrue(r['equilibrium_valid'])
        self.assertTrue(np.isfinite([r['trim_m'],r['GM_corrected_m']]).all())
        self.assertAlmostEqual(r['J_trim_abs_over_L'],abs(r['trim_m'])/80.,10)

    def test_companion_native_zone_map(self):
        # Minimal graph/companion stand-ins preserve spatial mapping exactly.
        from types import SimpleNamespace
        v=fixture();labels=v.labels
        node=np.arange(labels.size,dtype=np.int32).reshape(labels.shape)
        comp={'native_zone_id':node,'native_zone_labels':labels.ravel(),'native_occupied_fraction':v.hull_fraction}
        graph=SimpleNamespace(ship_type=3,L=v.L,B=v.B,D=v.D,T=5.,nz_hull=v.nz_hull)
        got=from_companion(graph,comp)
        np.testing.assert_array_equal(got.labels,labels)
        new=from_companion(graph,comp,labels_by_node=labels.ravel())
        self.assertEqual(compare_pair(got,new)['delta_KG_over_D'],0.)

    def test_actual_npzfile_adapter(self):
        v=fixture()
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'fixture.npz'
            nodes=np.arange(v.labels.size,dtype=np.int32).reshape(v.labels.shape)
            np.savez_compressed(path,native_zone_id=nodes,
                                native_occupied_fraction=v.hull_fraction,
                                native_zone_labels=v.labels.ravel())
            graph=SimpleNamespace(ship_type=3,L=v.L,B=v.B,D=v.D,T=5.,nz_hull=v.nz_hull)
            with np.load(path,allow_pickle=False) as data:
                got=from_companion(graph,data)
            np.testing.assert_array_equal(got.labels,v.labels)

    def test_validate_inconsistent_hull(self):
        v=fixture();a=v.hull_fraction.copy();a[0,0,0]=0.0
        with self.assertRaisesRegex(ValueError,'EMPTY cells'):
            evaluate_volume(EngineeringVolume(v.labels,a,v.family,v.L,v.B,v.D,v.nz_hull))

    def test_companion_rejects_invalid_labels_before_integer_cast(self):
        from types import SimpleNamespace
        v=fixture()
        comp={'native_zone_id':np.zeros(v.labels.shape,dtype=np.int16),
              'native_occupied_fraction':v.hull_fraction}
        graph=SimpleNamespace(ship_type=3,L=v.L,B=v.B,D=v.D,T=5.,nz_hull=v.nz_hull)
        for labels in (np.array([1.5]), np.array([65536]), np.array([9])):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError,'integer semantic labels'):
                from_companion(graph,comp,labels_by_node=labels)

    def test_companion_rejects_invalid_node_indices(self):
        from types import SimpleNamespace
        v=fixture()
        comp={'native_zone_id':np.full(v.labels.shape,-2,dtype=np.int16),
              'native_occupied_fraction':v.hull_fraction,'native_zone_labels':np.array([1])}
        graph=SimpleNamespace(ship_type=3,L=v.L,B=v.B,D=v.D,T=5.,nz_hull=v.nz_hull)
        with self.assertRaisesRegex(ValueError,'integer node indices'):
            from_companion(graph,comp)

    def test_pair_rejects_different_reference_drafts(self):
        from dataclasses import replace
        v=fixture()
        with self.assertRaisesRegex(ValueError,'ship particulars differ'):
            compare_pair(replace(v,T=4.),replace(v,T=5.))

if __name__=='__main__':unittest.main()
