import unittest
from mpilock import sync
import time
import mpi4py.MPI as mpi

rank = mpi.COMM_WORLD.Get_rank()


class TestLocking(unittest.TestCase):
    def setUp(self):
        self.controller = sync()
        self.controller._comm.Barrier()

    def tearDown(self):
        self.controller.close()
        self.controller._comm.Barrier()

    def test_controller_context(self):
        with self.controller as ctrl:
            self.assertEqual(self.controller, ctrl, "context return value")
        self.assertTrue(self.controller.closed, "context didn't close controller")

    def test_single_read_lock(self):
        if rank == 1:
            with self.controller.read():
                pass

    def test_concurrent_read_lock(self):
        t = time.time()
        with self.controller.read():
            time.sleep(1)
        self.assertAlmostEqual(
            1, time.time() - t, 1, "Concurrent read locks failed to run parallelly."
        )

    def test_single_read_lock(self):
        if rank == 1:
            with self.controller.write():
                pass

    def test_concurrent_write_lock(self):
        c = self.controller
        t = time.time()
        with c.write():
            time.sleep(0.1)
        c._comm.Barrier()
        self.assertAlmostEqual(
            0.1 * c._comm.Get_size(),
            time.time() - t,
            1,
            "Concurrent write locks failed to run serially.",
        )

    def test_collective_fence(self):
        spy = False
        with self.controller.single_write() as fence:
            fence.guard()
            spy = True
        if rank == 0:
            self.assertEqual(True, spy, "Master didn't enter collective write.")
        else:
            self.assertEqual(False, spy, "Non-master entered collective write.")

    def test_fence_collecting(self):
        with self.controller.single_write() as fence:
            fence.guard()
            fence.share(5)
        r = fence.collect()
        self.assertEqual(5, r, "Fenced resource was not correctly collected.")

    def test_collective_handle(self):
        essential = 5
        spy = False
        with self.controller.single_write(handle=essential) as handle:
            if handle is not None:
                spy = True
        if rank == 0:
            self.assertEqual(True, spy, "Master didn't utilize handle.")
        else:
            self.assertEqual(False, spy, "Non-master utilized handle.")
