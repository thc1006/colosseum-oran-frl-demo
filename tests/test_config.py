import unittest
from colosseum_oran_frl_demo.config import Paths, HP, hp_dict
from pathlib import Path

class TestConfig(unittest.TestCase):
    def test_paths(self):
        """Test if Paths are correct type and exist."""
        self.assertIsInstance(Paths.root, Path)
        self.assertTrue(Paths.raw_data.exists())
        self.assertTrue(Paths.processed.exists())

    def test_hyperparameters(self):
        """Test if hyperparameters are of correct type."""
        self.assertIsInstance(HP.lr, float)
        self.assertIsInstance(HP.gamma, float)
        self.assertIsInstance(HP.eps_decay, float)
        self.assertIsInstance(HP.local_steps, int)

    def test_hp_dict(self):
        d = hp_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d['lr'], HP.lr)

if __name__ == '__main__':
    unittest.main()
