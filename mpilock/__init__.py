__author__ = "Robin De Schepper"
__email__ = "robingilbert.deschepper@unipv.it"

__version__ = "2.1.0"

import mpi4py.MPI as MPI
import os
import sys
import time
import atexit
import threading
import warnings
import numpy as np
from opentelemetry import trace as _otel_trace

_tracer = _otel_trace.get_tracer("mpilock", __version__)


def sync(comm=None, master=0, pump=None, pump_interval=1e-4):
    """
    Create a :class:`.WindowController` that synchronizes read write operations across all
    MPI processes in the communicator.

    :param comm: MPI communicator
    :type comm: :class:`mpi4py.MPI.Communicator`
    :param master: Rank of the master of the communicator. All lock state lives in the
      master's MPI windows, so every acquire and release only ever needs cooperation
      from the master, never from the other (possibly busy) ranks.
    :type master: int
    :param pump: Run a background daemon thread on the master that keeps the MPI progress
      engine turning, so lock operations issued by other ranks (which target the master's
      windows by passive-target RMA) complete even while the master's main thread is busy
      with non-MPI work. Requires the MPI runtime to be initialized with
      ``MPI_THREAD_MULTIPLE``. Only the master rank starts a thread. Defaults to the
      ``MPILOCK_PUMP`` environment variable (on unless set to ``"0"``). Disable it when
      the runtime lacks ``MPI_THREAD_MULTIPLE`` and the master is kept responsive by other
      means.
    :type pump: bool
    :param pump_interval: Seconds the master's progress pump sleeps between MPI calls.
    :type pump_interval: float

    :return: A controller
    :rtype: :class:`.WindowController`
    """
    return WindowController(comm, master, pump=pump, pump_interval=pump_interval)


class WindowController:
    """
    The ``WindowController`` manages the state of the MPI windows underlying the lock
    functionality. Instances can be created using the :func:`.sync` factory function.

    The controller can create read and write locks during which your MPI processes are
    aware of each other's operations and a write lock will never be granted if other
    read or write operations are ongoing, while read locks may be granted while other read
    operations are ongoing, but not if any write locks are acquired or being requested.

    All lock state is centralized in the master rank's windows: a writer mutex window and
    a reader-count window. Acquiring or releasing any lock therefore only requires the
    master to make MPI progress, never the other ranks. Passive-target RMA only advances
    while the target is inside MPI, so the master runs a small daemon thread that keeps its
    progress engine turning (see the ``pump`` argument of :func:`.sync`); otherwise a busy
    master would stall every other rank's lock operations.
    """

    def __init__(self, comm=None, master=0, pump=None, pump_interval=1e-4):
        if pump is None:
            pump = os.environ.get("MPILOCK_PUMP", "1") != "0"
        if comm is None:
            comm = MPI.COMM_WORLD
        self._comm = comm
        self._size = comm.Get_size()
        self._rank = comm.Get_rank()
        self._master = master

        # Reader count (lives canonically in the master's window) and the writer mutex
        # window. The buffers on non-master ranks are unused; all RMA targets the master.
        self._count_buffer = np.zeros(1, dtype=np.int64)
        self._write_buffer = np.zeros(1, dtype=np.uint64)
        # A private window the master's progress pump operates on; touched by no one else.
        self._pump_buffer = np.zeros(1, dtype=np.uint64)
        self._count_window = self._window(self._count_buffer)
        self._write_window = self._window(self._write_buffer)
        self._pump_window = self._window(self._pump_buffer)
        # Re-entrant locks are tracked locally; only the outermost lock touches the master.
        self._read_depth = 0
        self._write_depth = 0
        self._closed = False

        self._pump_interval = pump_interval
        self._pump_thread = None
        self._pump_stop = None
        if pump and self._size > 1 and self._rank == self._master:
            if MPI.Query_thread() == MPI.THREAD_MULTIPLE:
                self._pump_stop = threading.Event()
                self._pump_thread = threading.Thread(
                    target=self._pump, name="mpilock-progress", daemon=True
                )
                self._pump_thread.start()
            else:
                warnings.warn(
                    "mpilock progress pump disabled: MPI runtime did not provide "
                    "MPI_THREAD_MULTIPLE. Lock acquisitions will stall whenever the "
                    "master is outside MPI. Initialize mpi4py with "
                    "`mpi4py.rc.thread_level = 'multiple'` against a thread-multiple "
                    "MPI build, or pass `pump=False` to silence this warning."
                )
        atexit.register(lambda: self.close())

    def _pump(self):
        # Keep the MPI progress engine turning so passive-target RMA from other ranks
        # against the master's windows (lock acquires and releases) completes while this
        # rank's main thread is busy with non-MPI work. A bare Iprobe drives the engine
        # too weakly to clear many concurrent acquisitions; a real RMA op (lock + get +
        # flush + unlock) on a private window pushes it hard enough that they complete
        # promptly. The window is owned by this pump alone, so locking it never contends
        # with the lock protocol's own windows.
        comm = self._comm
        stop = self._pump_stop
        interval = self._pump_interval
        m = self._master
        pw = self._pump_window
        dummy = np.zeros(1, dtype=np.uint64)
        while not stop.is_set():
            try:
                pw.Lock(m, MPI.LOCK_SHARED)
                pw.Get([dummy, MPI.UINT64_T], m)
                pw.Flush(m)
                pw.Unlock(m)
                comm.Iprobe(MPI.ANY_SOURCE, MPI.ANY_TAG)
            except Exception:
                break
            time.sleep(interval)

    @property
    def master(self):
        """
        Return the MPI rank of the master process.
        """
        return self._master

    @property
    def rank(self):
        """
        Return the MPI rank of this process.
        """
        return self._rank

    @property
    def closed(self):
        """
        Is this ``WindowController`` in a closed state? If so, further locks can not be
        requested.
        """
        return self._closed

    def close(self):
        """
        Close the ``WindowController``, stop the progress pump, and free its MPI Windows.
        """
        if self._closed:
            return
        self._closed = True
        if self._pump_thread is not None:
            self._pump_stop.set()
            self._pump_thread.join(timeout=1.0)
            self._pump_thread = None
        try:
            self._count_window.Free()
            self._write_window.Free()
            self._pump_window.Free()
        except MPI.Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _window(self, buffer):
        if self._comm.Get_size() == 1:
            return _WindowMock(buffer)
        return MPI.Win.Create(buffer, True, MPI.INFO_NULL, self._comm)

    def read(self):
        """
        Acquire a read lock. Read locks can be granted while other read locks are held,
        but will not start as long as write locks are held or being requested (write
        operations have priority over read operations).

        The preferred idiom for read locks is as follows:

        .. code-block:: python

            controller = sync()
            with controller.read():
                # Perform reading operation
                pass

        :return: A read lock
        """
        return _ReadLock(self)

    def write(self):
        """
        Acquire a write lock. Will wait for all active read locks to be released and
        prevent any new read locks from being aqcuired.

        The preferred idiom for write locks is as follows:

        .. code-block:: python

            controller = sync()
            with controller.write():
                # Perform writing operation
                pass

        Keep in mind that if you run this code on multiple processes at the same time that
        they will write one by one, but they will still all write eventually. If only one
        of the nodes needs to perform the writing operation see :meth:`~.WindowController.single_write`

        :return: An unfenced write lock
        """
        return _WriteLock(self)

    def single_write(self, handle=None, rank=None):
        """
        Perform a collective operation where only 1 node writes to the resource and the
        other processes wait for this operation to complete.

        Python does not support any long jump patterns so the preferred idiom for
        collective write locks is the fencing pattern:

        .. code-block:: python

            controller = sync()
            with controller.single_write() as fence:
                # Kick out any processes that don't have to write
                fence.guard()
                # Perform writing operation on just 1 process
                pass
            # All kicked out processes resume code together outside of the with block.

        :return: A fenced write lock.
        """
        if rank is None:
            rank = self._master
        fence = Fence(rank, self._rank == rank, self._comm)
        if self._rank == rank:
            return _WriteLock(self, fence=fence, handle=handle)
        elif handle:
            return _NoHandle(self._comm)
        else:
            return fence


class _WindowMock:
    def __init__(self, buffer):
        self._buffer = buffer

        def noop(*args, **kwargs):
            pass

        noops = [
            "Accumulate",
            "Flush",
            "Flush_all",
            "Free",
            "Get",
            "Lock",
            "Lock_all",
            "Unlock",
            "Unlock_all",
        ]
        for n in noops:
            setattr(self, n, noop)


class _ReadLock:
    def __init__(self, controller):
        self._ctrl = controller

    def __enter__(self):
        ctrl = self._ctrl
        nested = ctrl._read_depth > 0 or ctrl._write_depth > 0
        cm = _tracer.start_as_current_span(
            "mpilock.read",
            attributes={
                "mpi.rank": ctrl._rank,
                "mpi.master": ctrl._master,
                "mpilock.nested": nested,
            },
        )
        self._otel_span_ctx = cm
        cm.__enter__()
        # This instance owns the MPI registration only if it was the outermost
        # one; nested instances must not touch MPI on either enter or exit, or
        # the master's reader count goes out of sync with the actual readers.
        self._registered = not nested
        if nested:
            ctrl._read_depth += 1
        else:
            self._read_lock()
            ctrl._read_depth = 1

    def _read_lock(self):
        ctrl = self._ctrl
        w = ctrl._write_window
        c = ctrl._count_window
        m = ctrl._master
        with _tracer.start_as_current_span(
            "mpilock.read.wait", attributes={"mpi.rank": ctrl._rank}
        ):
            # Gate against writers: a writer holds this window exclusively for its whole
            # critical section, so taking it here blocks while a write is in progress and
            # gives writers priority.
            w.Lock(m)
            # MPI_Win_lock may return before the exclusive lock is acquired. A Get + Flush
            # forces the runtime to actually hold the lock before we proceed.
            _dummy = np.zeros(1, dtype=np.uint64)
            w.Get([_dummy, MPI.UINT64_T], m)
            w.Flush(m)
            # Register as an active reader by atomically bumping the master's count.
            c.Lock(m, MPI.LOCK_SHARED)
            one = np.ones(1, dtype=np.int64)
            c.Accumulate([one, MPI.INT64_T], m, op=MPI.SUM)
            c.Flush(m)
            c.Unlock(m)
            w.Unlock(m)

    def __exit__(self, exc_type, exc_value, traceback):
        ctrl = self._ctrl
        ctrl._read_depth -= 1
        if self._registered:
            self._read_unlock()
        self._otel_span_ctx.__exit__(exc_type, exc_value, traceback)

    def _read_unlock(self):
        ctrl = self._ctrl
        c = ctrl._count_window
        m = ctrl._master
        # Lockless decrement: the count lives in its own window, so a writer can hold the
        # writer-mutex window and still observe this drop while it waits for readers to
        # drain. A shared lock on the count window allows concurrent atomic accumulates.
        c.Lock(m, MPI.LOCK_SHARED)
        neg = np.array([-1], dtype=np.int64)
        c.Accumulate([neg, MPI.INT64_T], m, op=MPI.SUM)
        c.Flush(m)
        c.Unlock(m)


class _WriteLock:
    def __init__(self, controller, fence=None, handle=None):
        self._ctrl = controller
        self._fence = fence
        self._handle = handle

    def __enter__(self):
        ctrl = self._ctrl
        in_write = ctrl._write_depth > 0
        in_read = ctrl._read_depth > 0
        # A write inside an outer read cannot just acquire: the outer read
        # registered this rank as a reader, so the writer-mutex's spin on the
        # master's count would never see zero. Promote: drop the read
        # registration on enter, take the write exclusively, then re-register
        # as a reader on exit so the outer `with c.read():` exits cleanly.
        promoted = in_read and not in_write
        cm = _tracer.start_as_current_span(
            "mpilock.write",
            attributes={
                "mpi.rank": ctrl._rank,
                "mpi.master": ctrl._master,
                "mpilock.nested": in_write,
                "mpilock.promoted": promoted,
            },
        )
        self._otel_span_ctx = cm
        cm.__enter__()
        # An instance only releases what it acquired: only the outer write (or a
        # promoted write) holds the writer-mutex and must Unlock it on exit; only
        # a promoted write needs to re-register as a reader on exit.
        self._owns_window_lock = not in_write
        self._promoted_from_read = promoted
        if in_write:
            return self._nested_write_lock()
        if promoted:
            self._drop_reader_registration()
        return self._acquire_lock()

    def _acquire_lock(self):
        ctrl = self._ctrl
        w = ctrl._write_window
        c = ctrl._count_window
        m = ctrl._master
        with _tracer.start_as_current_span(
            "mpilock.write.wait", attributes={"mpi.rank": ctrl._rank}
        ):
            # Exclusive on the writer-mutex window: serializes writers and blocks new
            # readers (they take this same window to register), giving writers priority.
            # Held until __exit__.
            w.Lock(m)
            _dummy = np.zeros(1, dtype=np.uint64)
            w.Get([_dummy, MPI.UINT64_T], m)
            w.Flush(m)
            # Wait for active readers to drain. We hold the writer mutex, so no new readers
            # can register; readers only release (lockless decrement), so the count is
            # monotonically non-increasing here and the spin is guaranteed to terminate.
            c.Lock(m, MPI.LOCK_SHARED)
            val = np.zeros(1, dtype=np.int64)
            while True:
                c.Get([val, MPI.INT64_T], m)
                c.Flush(m)
                if val[0] == 0:
                    break
            c.Unlock(m)
        ctrl._write_depth = 1
        if self._handle is not None:
            return self._handle
        elif self._fence is not None:
            return self._fence

    def _nested_write_lock(self):
        self._ctrl._write_depth += 1
        if self._handle is not None:  # pragma: nocover
            return self._handle
        elif self._fence is not None:  # pragma: nocover
            return self._fence

    def _drop_reader_registration(self):
        # Mirror of _ReadLock._read_unlock: undo the outer reader's atomic
        # increment on the master's count so the writer-mutex spin can drain.
        ctrl = self._ctrl
        c = ctrl._count_window
        m = ctrl._master
        c.Lock(m, MPI.LOCK_SHARED)
        neg = np.array([-1], dtype=np.int64)
        c.Accumulate([neg, MPI.INT64_T], m, op=MPI.SUM)
        c.Flush(m)
        c.Unlock(m)

    def _restore_reader_registration(self):
        # Mirror of _ReadLock._read_lock: re-register this rank as a reader so
        # the outer read context can exit cleanly. The writer-mutex hand-off has
        # to go through the same w.Lock(m) gate readers use, or a new reader
        # could slip past during the race window.
        ctrl = self._ctrl
        w = ctrl._write_window
        c = ctrl._count_window
        m = ctrl._master
        w.Lock(m)
        _dummy = np.zeros(1, dtype=np.uint64)
        w.Get([_dummy, MPI.UINT64_T], m)
        w.Flush(m)
        c.Lock(m, MPI.LOCK_SHARED)
        one = np.ones(1, dtype=np.int64)
        c.Accumulate([one, MPI.INT64_T], m, op=MPI.SUM)
        c.Flush(m)
        c.Unlock(m)
        w.Unlock(m)

    def __exit__(self, exc_type, exc_value, traceback):
        ctrl = self._ctrl
        ctrl._write_depth -= 1
        if exc_type is not None:  # pragma: nocover
            warnings.warn(
                "Exception during write lock. Deadlock might occur if you use `.collect`."
            )
        if self._owns_window_lock:
            ctrl._write_window.Unlock(ctrl._master)
            if self._fence is not None:
                self._fence._comm.Barrier()
                sys.stderr.flush()
        if self._promoted_from_read:
            self._restore_reader_registration()
        self._otel_span_ctx.__exit__(exc_type, exc_value, traceback)


class Fence:
    """
    Can be used to fence off pieces of code from processes that shouldn't access it.
    Additionally it can be used to share a resource to all processes that was created
    within the fenced off code block using :meth:`.Fence.share` and
    :meth:`.Fence.collect`.
    """

    def __init__(self, master, access, comm):
        """
        Create a fence to guard code blocks from certain MPI processes within a try or
        with statement.

        :param master: MPI rank that controls the fence and any resource sharing.
        :type master: int
        :param access: May this MPI process enter the fenced off block?
        :type access: bool
        :param comm: MPI communicator for collective resource sharing.
        """
        self._master = master
        self._access = access
        self._comm = comm
        self._obj = None

    def guard(self):
        """
        Kicks out all MPI processes that do not have access to the fenced off code block.
        Works only within a ``with`` statement or a ``try`` statement that catches
        :class:`.FencedSignal` exceptions.
        """
        if not self._access:
            raise FencedSignal()

    def share(self, obj):
        """
        Put an object to share with all other MPI processes from within a fenced off code
        block.
        """
        self._obj = obj

    def collect(self):
        """
        Collect the object that was put to share within the fenced off code block.

        :return: Shared object
        :rtype: any
        """
        return self._comm.bcast(self._obj, root=self._master)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._comm.Barrier()
        if exc_type is FencedSignal:
            return True


class _NoHandle:
    def __init__(self, comm):
        self._comm = comm

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        self._comm.Barrier()


class FencedSignal(Exception):
    pass


__all__ = ["sync", "WindowController", "Fence", "FencedSignal"]
