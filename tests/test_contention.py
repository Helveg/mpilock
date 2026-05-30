"""
Multi-rank contention and race condition tests for mpilock.

These tests verify that read and write locks correctly serialize access,
prevent data races, and remain deadlock-free under concurrent workloads.
They require at least 2 MPI ranks to be meaningful and are automatically
skipped when run serially.

Important: All MPI collective operations (Barrier, bcast, write/read locks)
must complete on *every* rank before any assertion is raised.  If a rank
raises AssertionError while others are still waiting at a collective call,
the program deadlocks.  The pattern used throughout is therefore:

    1. Run all MPI-collective work (locks, sleeps, file I/O)
    2. c._comm.Barrier()          # synchronize before asserting
    3. assertions (safe to raise)
    4. optional cleanup + c._comm.Barrier()

Run with:
    mpiexec --oversubscribe -n 4 python -m unittest tests/test_contention.py
"""

import os
import tempfile
import time
import unittest

import mpi4py.MPI as mpi
import numpy as np

from mpilock import sync

rank = mpi.COMM_WORLD.Get_rank()
size = mpi.COMM_WORLD.Get_size()

# Convenience decorator: skip tests that need real MPI parallelism.
_multi_rank = unittest.skipIf(size < 2, "requires at least 2 MPI ranks")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _broadcast_tmpfile(comm):
    """Rank 0 creates a temp .npy file; all ranks receive its path."""
    tmpfile = None
    if rank == 0:
        fd, tmpfile = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
    return comm.bcast(tmpfile, root=0)


def _remove_tmpfile(tmpfile, comm):
    """Rank 0 removes the temp file after all ranks are done reading it."""
    comm.Barrier()
    if rank == 0:
        os.unlink(tmpfile)
    comm.Barrier()


# ---------------------------------------------------------------------------
# Read / write mutual exclusion
# ---------------------------------------------------------------------------


class TestReadWriteContention(unittest.TestCase):
    """Write locks must block reads, reads must block writes, and concurrent
    reads must never be serialized by one another."""

    def setUp(self):
        self.c = sync()
        self.c._comm.Barrier()

    def tearDown(self):
        self.c.close()
        self.c._comm.Barrier()

    @_multi_rank
    def test_write_blocks_reads(self):
        """A held write lock must prevent other ranks from acquiring read locks."""
        c = self.c
        elapsed = None
        if rank == 0:
            with c.write():
                time.sleep(0.4)
        else:
            time.sleep(0.05)  # let rank 0 acquire the write lock first
            t = time.time()
            with c.read():
                pass
            elapsed = time.time() - t
        c._comm.Barrier()
        if rank != 0:
            self.assertGreater(
                elapsed, 0.3, "Read lock was acquired while a write lock was still held"
            )

    @_multi_rank
    def test_reads_block_write(self):
        """Active read locks must prevent write lock acquisition."""
        c = self.c
        elapsed = None
        if rank == 0:
            with c.read():
                time.sleep(0.4)
        else:
            time.sleep(0.05)  # let rank 0 acquire the read lock first
            t = time.time()
            with c.write():
                pass
            elapsed = time.time() - t
        c._comm.Barrier()
        if rank != 0:
            self.assertGreater(
                elapsed, 0.3, "Write lock was acquired while a read lock was still held"
            )

    @_multi_rank
    def test_concurrent_reads_not_serialized(self):
        """Multiple ranks holding read locks simultaneously must not block each other."""
        from mpi4py import MPI

        c = self.c
        # All ranks take a read lock and sleep for 4 s. If reads were
        # serialized the total would be 4 * size; concurrently it is ~4 s.
        t = time.time()
        with c.read():
            for i in range(1000):
                MPI.COMM_WORLD.Iprobe()
                time.sleep(0.004)
        elapsed = time.time() - t
        c._comm.Barrier()
        self.assertLess(
            elapsed,
            6.0,
            f"Read locks appear to have serialized: elapsed {elapsed:.2f}s, "
            f"expected ~4s concurrent across {size} ranks",
        )

    @_multi_rank
    def test_reads_run_concurrently_after_write_releases(self):
        """Once a write lock releases, all queued reads must proceed in parallel."""
        c = self.c
        elapsed = None
        if rank == 0:
            with c.write():
                time.sleep(0.4)
        else:
            time.sleep(0.05)  # queue up while write is active
            t = time.time()
            with c.read():
                time.sleep(0.3)  # all non-0 ranks sleep concurrently
            elapsed = time.time() - t
        c._comm.Barrier()
        if rank != 0:
            # Waited ~0.35 s for write, then read 0.3 s concurrently.
            # If reads were serialized, elapsed would grow with (size - 1).
            self.assertGreater(elapsed, 0.55, "Read did not wait for write to finish")
            self.assertLess(
                elapsed,
                0.35 + 0.3 + 0.3,  # generous for slow machines
                "Post-write reads were serialized instead of running concurrently",
            )


# ---------------------------------------------------------------------------
# Data-integrity under concurrent access
# ---------------------------------------------------------------------------


class TestDataIntegrity(unittest.TestCase):
    """Lock semantics must prevent observable data races on shared state."""

    def setUp(self):
        self.c = sync()
        self.c._comm.Barrier()

    def tearDown(self):
        self.c.close()
        self.c._comm.Barrier()

    @_multi_rank
    def test_write_lock_prevents_lost_updates(self):
        """Write lock must make read-modify-write cycles on a shared file atomic."""
        c = self.c
        tmpfile = _broadcast_tmpfile(c._comm)
        if rank == 0:
            np.save(tmpfile, np.zeros(1, dtype=np.int64))
        c._comm.Barrier()

        iterations = 5
        for _ in range(iterations):
            with c.write():
                val = np.load(tmpfile)
                val[0] += 1
                np.save(tmpfile, val)
        c._comm.Barrier()

        result = int(np.load(tmpfile)[0])
        expected = size * iterations
        _remove_tmpfile(tmpfile, c._comm)
        self.assertEqual(
            expected,
            result,
            f"Lost-update race detected: expected counter={expected}, got {result}",
        )

    @_multi_rank
    def test_write_lock_exclusive_during_hold(self):
        """While any rank holds the write lock, no other rank may modify shared state."""
        c = self.c
        tmpfile = _broadcast_tmpfile(c._comm)
        if rank == 0:
            np.save(tmpfile, np.zeros(size, dtype=np.int64))
        c._comm.Barrier()

        # Each rank marks its own slot, holds the lock briefly, then re-reads
        # to verify no concurrent writer modified the file during the hold.
        was_clobbered = False
        with c.write():
            arr = np.load(tmpfile)
            arr[rank] = rank + 1
            np.save(tmpfile, arr)
            time.sleep(0.05)  # hold the lock while sleeping
            arr_now = np.load(tmpfile)
            was_clobbered = int(arr_now[rank]) != rank + 1
        c._comm.Barrier()

        final = np.load(tmpfile)
        _remove_tmpfile(tmpfile, c._comm)

        self.assertFalse(
            was_clobbered,
            "Shared file was modified by another rank while we held the write lock",
        )
        for r in range(size):
            self.assertEqual(
                r + 1,
                int(final[r]),
                f"Rank {r}'s write was lost or corrupted in the final state",
            )

    @_multi_rank
    def test_mixed_read_write_workload(self):
        """Even ranks read, odd ranks write; final counter must equal total write count."""
        c = self.c
        tmpfile = _broadcast_tmpfile(c._comm)
        if rank == 0:
            np.save(tmpfile, np.zeros(1, dtype=np.int64))
        c._comm.Barrier()

        iterations = 10
        for _ in range(iterations):
            if rank % 2 == 0:
                with c.read():
                    np.load(tmpfile)  # read-only; value not asserted here
            else:
                with c.write():
                    val = np.load(tmpfile)
                    val[0] += 1
                    np.save(tmpfile, val)
        c._comm.Barrier()

        odd_count = sum(1 for r in range(size) if r % 2 != 0)
        expected = odd_count * iterations
        result = int(np.load(tmpfile)[0])
        _remove_tmpfile(tmpfile, c._comm)
        self.assertEqual(
            expected,
            result,
            f"Mixed workload: expected {expected} increments, got {result}",
        )


# ---------------------------------------------------------------------------
# Nested locks under contention
# ---------------------------------------------------------------------------


class TestNestedUnderContention(unittest.TestCase):
    """Nested lock combinations must not deadlock when other ranks compete."""

    def setUp(self):
        self.c = sync()
        self.c._comm.Barrier()

    def tearDown(self):
        self.c.close()
        self.c._comm.Barrier()

    @_multi_rank
    def test_nested_write_read_blocks_competing_writes(self):
        """Nested read-inside-write must block competing writes for its full duration."""
        c = self.c
        elapsed = None
        if rank == 0:
            with c.write():
                # Nested read uses the fast path (no MPI Window call needed)
                # because locked() returns True — must not deadlock.
                with c.read():
                    time.sleep(0.3)
        else:
            time.sleep(0.05)
            t = time.time()
            with c.write():
                pass
            elapsed = time.time() - t
        c._comm.Barrier()
        if rank != 0:
            self.assertGreater(
                elapsed,
                0.2,
                "Competing write acquired lock before nested write+read completed",
            )

    @_multi_rank
    def test_nested_read_write_blocks_competing_reads(self):
        """Nested write-inside-read must block competing reads for its full duration."""
        c = self.c
        elapsed = None
        if rank == 0:
            with c.read():
                with c.write():
                    time.sleep(0.3)
        else:
            time.sleep(0.05)
            t = time.time()
            with c.read():
                pass
            elapsed = time.time() - t
        c._comm.Barrier()
        if rank != 0:
            self.assertGreater(
                elapsed,
                0.2,
                "Competing read acquired lock before nested read+write completed",
            )

    @_multi_rank
    def test_deep_nesting_under_write_pressure(self):
        """Deeply nested write→read→write must not deadlock under write pressure."""
        c = self.c
        elapsed = None
        if rank == 0:
            with c.write():
                with c.read():
                    with c.write():
                        time.sleep(0.3)
        else:
            time.sleep(0.05)
            t = time.time()
            with c.write():
                pass
            elapsed = time.time() - t
        c._comm.Barrier()
        if rank != 0:
            self.assertGreater(
                elapsed,
                0.2,
                "Write acquired lock before deeply nested sequence completed",
            )

    @_multi_rank
    def test_read_inside_write_does_not_release_master_count(self):
        """A read nested inside a write must not touch the master's reader count
        on enter or exit. If the inner read's exit calls _read_unlock the count
        goes negative, and the next writer spins on val != 0 forever."""
        c = self.c
        elapsed = None
        if rank == 0:
            with c.write():
                with c.read():
                    pass
                with c.read():
                    pass
        else:
            time.sleep(0.05)
            t = time.time()
            with c.write():
                pass
            elapsed = time.time() - t
        c._comm.Barrier()
        if rank != 0:
            self.assertLess(
                elapsed,
                1.0,
                "Competing write stalled after rank 0 unwound nested reads "
                "inside a write: the master's reader count is over-released.",
            )

    @_multi_rank
    def test_write_inside_read_blocks_competing_writes(self):
        """A write nested inside an outer read must hold the writer-mutex for
        its full duration. If promotion does not happen, the writer-mutex spin
        on the master's count would never succeed (this rank's outer read keeps
        the count above zero), so the test would either deadlock or, if the
        promotion is wrong in the other direction, competing writers would slip
        in before the inner write's sleep completes."""
        c = self.c
        elapsed = None
        if rank == 0:
            with c.read():
                with c.write():
                    time.sleep(0.3)
        else:
            time.sleep(0.05)
            t = time.time()
            with c.write():
                pass
            elapsed = time.time() - t
        c._comm.Barrier()
        if rank != 0:
            self.assertGreater(
                elapsed,
                0.2,
                "Competing write acquired the writer-mutex before the inner "
                "write inside an outer read completed: promotion did not hold "
                "the writer-mutex exclusively.",
            )

    @_multi_rank
    def test_write_inside_read_restores_reader_after_inner_write_exits(self):
        """After a promoted write inside an outer read exits, this rank must be
        re-registered as a reader so the outer read still blocks other writers
        for the rest of its duration."""
        c = self.c
        elapsed = None
        if rank == 0:
            with c.read():
                with c.write():
                    pass
                time.sleep(0.3)
        else:
            time.sleep(0.05)
            t = time.time()
            with c.write():
                pass
            elapsed = time.time() - t
        c._comm.Barrier()
        if rank != 0:
            self.assertGreater(
                elapsed,
                0.2,
                "Competing write acquired during the outer read's remaining "
                "hold: the inner write did not re-register this rank as a "
                "reader on exit.",
            )

    @_multi_rank
    def test_alternating_nested_patterns_under_contention(self):
        """Cycle through every nested pattern. If any combination corrupts the
        master's count or orphans the writer-mutex, the post-cycle acquires
        from other ranks stall instead of completing promptly."""
        c = self.c
        elapsed = None
        if rank == 0:
            with c.read():
                with c.read():
                    pass
            with c.write():
                with c.write():
                    pass
            with c.write():
                with c.read():
                    pass
            with c.read():
                with c.write():
                    pass
            with c.write():
                with c.read():
                    with c.write():
                        pass
        else:
            time.sleep(0.2)
            t = time.time()
            with c.write():
                pass
            with c.read():
                pass
            elapsed = time.time() - t
        c._comm.Barrier()
        if rank != 0:
            self.assertLess(
                elapsed,
                1.0,
                "Acquisitions after the nested-pattern cycle stalled: master "
                "state is corrupted.",
            )


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------


class TestStressLocking(unittest.TestCase):
    """High-frequency and mixed-pattern lock cycling must stay deadlock-free."""

    def setUp(self):
        self.c = sync()
        self.c._comm.Barrier()

    def tearDown(self):
        self.c.close()
        self.c._comm.Barrier()

    def test_rapid_alternating_read_write(self):
        """Rapid alternating reads and writes from all ranks must not deadlock."""
        c = self.c
        for i in range(30):
            if i % 3 == 0:
                with c.write():
                    pass
            else:
                with c.read():
                    pass
        c._comm.Barrier()

    def test_repeated_nested_write_read_cycles(self):
        """Nested write→read repeated many times must remain deadlock-free."""
        c = self.c
        for _ in range(20):
            with c.write():
                with c.read():
                    pass
        c._comm.Barrier()

    def test_repeated_nested_read_write_cycles(self):
        """Nested read→write repeated many times must remain deadlock-free."""
        c = self.c
        for _ in range(20):
            with c.read():
                with c.write():
                    pass
        c._comm.Barrier()

    @_multi_rank
    def test_write_data_integrity_under_stress(self):
        """Counter incremented under write lock during rapid cycling must be exact."""
        c = self.c
        tmpfile = _broadcast_tmpfile(c._comm)
        if rank == 0:
            np.save(tmpfile, np.zeros(1, dtype=np.int64))
        c._comm.Barrier()

        iterations = 20
        for i in range(iterations):
            if i % 4 == 0:
                with c.write():
                    val = np.load(tmpfile)
                    val[0] += 1
                    np.save(tmpfile, val)
            else:
                with c.read():
                    pass
        c._comm.Barrier()

        result = int(np.load(tmpfile)[0])
        expected = size * (iterations // 4)
        _remove_tmpfile(tmpfile, c._comm)
        self.assertEqual(
            expected,
            result,
            f"Stress integrity: expected {expected} increments, got {result}",
        )
