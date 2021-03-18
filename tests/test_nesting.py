import unittest
from mpilock import sync
import time
import mpi4py.MPI as mpi

rank = mpi.COMM_WORLD.Get_rank()


class TestNestedLocks(unittest.TestCase):
    def setUp(self):
        self.controller = sync()
        self.controller._comm.Barrier()

    def tearDown(self):
        self.controller.close()
        self.controller._comm.Barrier()

    def test_double_read(self):
        with self.controller.read() as ctrl:
            with self.controller.read() as ctrl:
                pass

    def test_double_write(self):
        with self.controller.write() as ctrl:
            with self.controller.write() as ctrl:
                pass

    def test_read_write(self):
        with self.controller.read() as ctrl:
            with self.controller.write() as ctrl:
                pass

    def test_write_read(self):
        with self.controller.write() as ctrl:
            with self.controller.read() as ctrl:
                pass
