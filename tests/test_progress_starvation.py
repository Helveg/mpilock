"""
Regression test for MPI passive-target progress starvation.

All lock state lives on the master and the master runs a progress pump, so
acquiring or releasing a lock only needs the master to make MPI progress, never
the other ranks. When ranks spend long stretches in non-MPI work between lock
operations, lock acquisition must therefore still complete promptly.

A design where a writer polls every rank's read flag (``Flush_all`` over all
ranks) instead stalls until the compute-bound ranks next re-enter MPI, so its
acquisition time scales with the off-MPI compute. These tests pin acquisition
time well below the compute window, so they fail on such a design and pass on the
centralized-master one.

Run with:
    mpiexec --oversubscribe -n 4 python -m unittest tests/test_progress_starvation.py
"""

import time
import unittest

import mpi4py.MPI as mpi

from mpilock import sync

rank = mpi.COMM_WORLD.Get_rank()
size = mpi.COMM_WORLD.Get_size()


def _busy_until(t_end):
    # Pure-Python busy work that makes no MPI calls, so it does not service the
    # progress engine. This is what starves a design that needs every rank to poll.
    x = 0
    while time.time() < t_end:
        for _ in range(50000):
            x += 1
    return x


class TestProgressStarvation(unittest.TestCase):
    def setUp(self):
        self.c = sync()
        self.c._comm.Barrier()

    def tearDown(self):
        self.c.close()
        self.c._comm.Barrier()

    @unittest.skipIf(size < 3, "needs >=3 ranks: a master, a prober, and a busy rank")
    def test_writer_not_starved_by_busy_ranks(self):
        """A writer must acquire promptly even while other ranks sit in long non-MPI
        compute. Its acquisition may only depend on the master, not on the busy ranks."""
        c = self.c
        duration = 2.0
        c._comm.Barrier()
        t_end = time.time() + duration
        max_acq = 0.0
        if rank == 0:
            # Master stays responsive (inside MPI) for the duration.
            while time.time() < t_end:
                mpi.COMM_WORLD.Iprobe(mpi.ANY_SOURCE, mpi.ANY_TAG)
                time.sleep(0.001)
        elif rank == 1:
            # Prober repeatedly acquires the write lock and records the worst acquire.
            while time.time() < t_end:
                t = time.perf_counter()
                with c.write():
                    pass
                max_acq = max(max_acq, time.perf_counter() - t)
        else:
            # Busy ranks: a long stretch of non-MPI compute, no lock, no MPI calls.
            _busy_until(t_end)
        global_max = c._comm.allreduce(max_acq, op=mpi.MAX)
        c._comm.Barrier()
        self.assertLess(
            global_max,
            0.5,
            f"Writer acquisition starved by compute-bound ranks "
            f"(worst acquire {global_max:.2f}s over a {duration}s window): acquisition "
            f"is waiting on busy workers instead of only the master.",
        )

    @unittest.skipIf(size < 2, "requires at least 2 MPI ranks")
    def test_acquisition_bounded_under_offmpi_compute(self):
        """With every rank doing a chunk of non-MPI compute between lock operations,
        acquisition must stay far below the compute-chunk duration."""
        c = self.c
        compute_s = 0.5
        iters = 6
        is_writer = rank % 4 == 0
        max_acq = 0.0
        c._comm.Barrier()
        for _ in range(iters):
            end = time.time() + compute_s
            _busy_until(end)  # spend time outside MPI, like the bender does
            t = time.perf_counter()
            if is_writer:
                with c.write():
                    pass
            else:
                with c.read():
                    pass
            max_acq = max(max_acq, time.perf_counter() - t)
        global_max = c._comm.allreduce(max_acq, op=mpi.MAX)
        c._comm.Barrier()
        self.assertLess(
            global_max,
            compute_s * 0.5,
            f"Lock acquisition scaled with off-MPI compute "
            f"(worst acquire {global_max:.2f}s, compute chunk {compute_s}s).",
        )
