"""
Regression test for MPI passive-target progress starvation.

All lock state lives on the master and the master runs a progress pump, so
acquiring or releasing a lock only needs the master to make MPI progress, never
the other ranks. When ranks spend long stretches in non-MPI work between lock
operations, lock acquisition must therefore still complete promptly.

A design where a writer polls every rank's read flag (``Flush_all`` over all
ranks) instead stalls until the compute-bound ranks next re-enter MPI, so its
acquisition time scales with the off-MPI compute. These tests bound acquisition
time far below the seconds-to-tens-of-seconds stall such a design exhibits at
scale, so they fail on it and pass on the centralized-master one.

The stall only manifests on an MPI setup where passive-target RMA depends on the
target making progress (network RMA, multi-node, no async progress thread; e.g.
CINECA g100). On a single-node, shared-memory MPI it does not appear, so these
tests pass on old and new code alike there. To reproduce it on a single node with
OpenMPI, force RMA over point-to-point (which needs target progress) and a thread
level it supports:

    OMPI_MCA_osc=pt2pt MPI4PY_RC_THREAD_LEVEL=single MPILOCK_PUMP=0 \\
        mpiexec --oversubscribe -n 4 python -m unittest tests/test_progress_starvation.py

Under that env the writer test fails on the old design and passes on the new one
(the master stays responsive, so centralizing state on it is enough). The pump
test needs MPI_THREAD_MULTIPLE (which pt2pt lacks), so it is skipped there.

Run normally with:
    mpiexec --oversubscribe -n 4 python -m unittest tests/test_progress_starvation.py
"""

import os
import time
import unittest

import mpi4py.MPI as mpi

from mpilock import sync

rank = mpi.COMM_WORLD.Get_rank()
size = mpi.COMM_WORLD.Get_size()

# The pump-dependent test only makes sense where the progress pump can actually run.
_pump_capable = (
    mpi.Query_thread() == mpi.THREAD_MULTIPLE
    and os.environ.get("MPILOCK_PUMP", "1") != "0"
)


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
    @unittest.skipUnless(
        _pump_capable, "needs the progress pump (MPI_THREAD_MULTIPLE and MPILOCK_PUMP)"
    )
    def test_acquisition_bounded_under_offmpi_compute(self):
        """With every rank doing a chunk of non-MPI compute between lock operations,
        acquisition must stay far below the compute-chunk duration. Unlike the writer
        test, no rank stays responsive here, so this relies on the master's pump."""
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
        # A starved design (per-rank polling, or no/weak pump) makes acquisition scale
        # with the off-MPI compute and the rank count: seconds to tens of seconds on
        # g100. The centralized master with a real-RMA progress pump keeps it well under
        # a second even with the master itself compute-bound (its pump thread fights the
        # GIL), so a sub-second bound passes the fix yet fails every starved variant.
        bound = 1.0
        self.assertLess(
            global_max,
            bound,
            f"Lock acquisition is not bounded under off-MPI compute "
            f"(worst acquire {global_max:.2f}s, bound {bound}s, compute chunk "
            f"{compute_s}s): the master's progress pump is not clearing acquisitions.",
        )
