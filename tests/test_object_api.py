import unittest
from mpilock import sync
import time
import mpi4py.MPI as mpi

rank = mpi.COMM_WORLD.Get_rank()


class TestWindowController(unittest.TestCase):
    def setUp(self):
        self.controller = sync()
        self.controller._comm.Barrier()

    def tearDown(self):
        self.controller.close()
        self.controller._comm.Barrier()

    def test_rank(self):
        self.assertEqual(mpi.COMM_WORLD.Get_rank(), self.controller.rank, "rank")

    def test_master(self):
        self.assertEqual(0, self.controller.master, "master")
        with sync(master=1) as ctrl:
            self.assertEqual(1, ctrl.master, "non default master")

    def test_closed(self):
        self.assertFalse(self.controller.closed, "closed")
        with self.controller as ctrl:
            self.assertFalse(ctrl.closed, "closed")
        self.assertTrue(ctrl.closed, "closed")
