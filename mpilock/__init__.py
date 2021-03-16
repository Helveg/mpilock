__author__ = "Robin De Schepper"
__email__ = "robingilbert.deschepper@unipv.it"

__version__ = "0.0.2"

import mpi4py.MPI as MPI
import atexit
import numpy as np, time


def sync(comm=None, master=0):
    """
    Create a :class:`.WindowController` that synchronizes read write operations across all
    MPI processes in the communicator.

    :param comm: MPI communicator
    :type comm: :class:`mpi4py.MPI.Communicator`
    :param master: Rank of the master of the communicator, will be picked whenever
      something needs to be organized or decided by a single node in the communicator.
    :type comm: int

    :return: A controller
    :rtype: :class:`.WindowController`
    """
    return WindowController(comm, master)


class WindowController:
    """
    The ``WindowController`` manages the state of the MPI windows underlying the lock
    functionality. Instances can be created using the :func:`.sync` factory function.

    The controller can create read and write locks during which your MPI processes are
    aware of each other's operations and a write lock will never be granted if other
    read or write operations are ongoing, while read locks may be granted while other read
    operations are ongoing, but not if any write locks are acquired or being requested.
    """

    def __init__(self, comm=None, master=0):
        if comm is None:
            comm = MPI.COMM_WORLD
        self._comm = comm
        self._size = comm.Get_size()
        self._rank = comm.Get_rank()
        self._master = master

        self._read_buffer = np.zeros(1, dtype=np.uint64)
        self._write_buffer = np.zeros(1, dtype=np.bool_)
        self._read_window = self._window(self._read_buffer)
        self._write_window = self._window(self._write_buffer)
        atexit.register(lambda: self.close())
        self._closed = False

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
        Close the ``WindowController`` and its underlying MPI Windows.
        """
        try:
            self._read_window.Free()
            self._write_window.Free()
        except MPI.Exception:
            pass
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _window(self, buffer):
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
        return _ReadLock(self._read_buffer, self._write_window, self._master)

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
        of the nodes needs to perform the writing operation see :method:`.WindowController.single_write`

        :return: An unfenced write lock
        """
        return _WriteLock(
            self._read_buffer,
            self._read_window,
            self._size,
            self._write_window,
            self._master,
        )

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
        fence = _Fence(self._rank == rank, self._comm)
        if self._rank == rank:
            return _WriteLock(
                self._read_buffer,
                self._read_window,
                self._size,
                self._write_window,
                self._master,
                fence=fence,
                handle=handle,
            )
        elif handle:
            return _NoHandle(self._comm)
        else:
            return fence


class _ReadLock:
    def __init__(self, read_buffer, write_window, root):
        self._read_buffer = read_buffer
        self._write_window = write_window
        self._root = root

    def __enter__(self):
        # Wait for the write lock to be available before starting your read operation
        self._write_window.Lock(self._root)
        self._read_buffer[0] = True
        self._write_window.Unlock(self._root)

    def __exit__(self, exc_type, exc_value, traceback):
        # Stop read operation any time
        self._read_buffer[0] = False


class _WriteLock:
    def __init__(
        self,
        read_buffer,
        read_window,
        size,
        write_window,
        root,
        fence=None,
        handle=None,
    ):
        self._read_buffer = read_buffer
        self._read_window = read_window
        self._size = size
        self._write_window = write_window
        self._root = root
        self._fence = fence
        self._handle = handle

    def __enter__(self):
        # A write lock can be opened on top of a read lock, but if our read
        # flag is `True` we'll be deadlocked waiting for it to be unset, so we
        # save our read state, unset it and instead restore it later.
        reading = self._read_buffer[0]
        self._write_window.Lock(0)
        self._read_buffer[0] = 0
        self._read_window.Lock_all()
        all_read = [np.zeros(1, dtype=np.uint64) for _ in range(self._size)]
        while True:
            for i in range(self._size):
                self._read_window.Get([all_read[i], MPI.BOOL], i)
            if sum(all_read)[0] == 0:
                break
        self._read_buffer[0] = reading
        self._read_window.Unlock_all()
        if self._handle is not None:
            return self._handle
        elif self._fence is not None:
            return self._fence

    def __exit__(self, exc_type, exc_value, traceback):
        self._write_window.Unlock(0)
        if self._fence is not None:
            self._fence._comm.Barrier()


class _Fence:
    def __init__(self, access, comm):
        self._access = access
        self._comm = comm

    def guard(self):
        if not self._access:
            raise _FencedSignal()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._comm.Barrier()
        if exc_type is _FencedSignal:
            return True


class _NoHandle:
    def __init__(self, comm):
        self._comm = comm

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        self._comm.Barrier()


class _FencedSignal(Exception):
    pass


__all__ = ["sync", "WindowController"]
