import unittest
from colosseum_oran_frl_demo.config import Paths, HP, hp_dict
import os

class TestConfig(unittest.TestCase):
    def test_paths(self):
        print(f"Root path: {Paths.ROOT}, exists: {Paths.ROOT.exists()}")
        print(f"Raw data path: {Paths.RAW_DATA}, exists: {Paths.RAW_DATA.exists()}")
        print(f"Processed path: {Paths.PROCESSED}, exists: {Paths.PROCESSED.exists()}")
        print(f"Outputs path: {Paths.OUTPUTS}, exists: {Paths.OUTPUTS.exists()}")
        self.assertTrue(Paths.ROOT.is_dir())
        self.assertTrue(Paths.RAW_DATA.is_dir())
        self.assertTrue(Paths.PROCESSED.is_dir())
        self.assertTrue(Paths.OUTPUTS.is_dir())

    def test_hyperparameters(self):
        self.assertIsInstance(HP.LR, float)
        self.assertIsInstance(HP.GAMMA, float)
        self.assertIsInstance(HP.EPS_DECAY, float)
        self.assertIsInstance(HP.LOCAL_STEPS, int)

    def test_hp_dict(self):
        d = hp_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d['LR'], HP.LR)

if __name__ == '__main__':
    unittest.main()
