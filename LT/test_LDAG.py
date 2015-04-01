
import unittest
from LDAG import tsort
import networkx as nx

class TestTopologicalSort(unittest.TestCase):
    def test_sorting(self):
        D = nx.DiGraph()
        D.add_edge(1, 0, weight=1)
        D.add_edge(3, 0, weight=1)
        D.add_edge(3, 1, weight=1)
        D.add_edge(4, 1, weight=1)
        D.add_edge(4, 3, weight=1)
        D.add_edge(5, 3, weight=1)

        self.assertEqual(tsort(D, 0), [0,1,3,4,5])

if __name__ == '__main__':
    unittest.main(verbosity=2)

