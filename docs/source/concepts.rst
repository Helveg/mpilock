Concepts
========

``mpilock`` is a reader-writer lock over MPI passive-target RMA. This page
describes the semantics it offers, where its state lives, and the requirements
that follow from doing concurrent MPI from a thread.

Reader-writer semantics
-----------------------

A :class:`.WindowController` synchronizes parallel access to a shared resource
across all MPI processes in a communicator. Two lock types are available:

* :meth:`~.WindowController.read` may be held by any number of processes
  concurrently.
* :meth:`~.WindowController.write` is exclusive. It blocks until no other
  process is holding a read or write lock.

Writes have priority over reads. Once a writer's acquire is in progress, new
read acquires block, so a continuous stream of readers cannot starve a writer.
This matches the standard "writer-preference" reader-writer lock semantics.

Re-entrant acquires (a ``with`` block nested inside another, on the same
controller) are tracked locally. Only the outermost acquire and release touch
MPI, so nesting has negligible overhead.

Where lock state lives
----------------------

All lock state is centralized in two MPI windows owned by the master rank: a
writer-mutex window and a reader-count window. Every acquisition's RMA
therefore targets the master:

* A writer takes the writer-mutex window exclusively, then spins on the
  master's reader count until it reaches zero.
* A reader registers under the writer-mutex window with an atomic increment on
  the master's count, then releases with a lockless atomic decrement on the
  reader-count window.

Worker ranks are never polled for lock state. They can spend arbitrarily long
stretches in non-MPI work between lock operations without stalling another
rank's acquisition.

The progress pump
-----------------

Passive-target RMA only advances while the *target* of an operation is inside
MPI. With state centralized on the master, every acquisition's RMA targets the
master, so the master must keep its MPI progress engine turning even while its
main thread is busy with non-MPI work.

``mpilock`` does this with a single daemon thread on the master rank that
performs a real RMA cycle (``Lock`` + ``Get`` + ``Flush`` + ``Unlock``) on a
private window every ``pump_interval`` seconds (default ``1e-4``). A bare
``Iprobe`` drives the progress engine too weakly to clear many concurrent
acquisitions; a real RMA operation pushes it hard enough that acquisitions
complete promptly.

The pump is enabled by default and configurable per controller via
:func:`.sync` (or :class:`.WindowController`). Process-wide it can be disabled
with the ``MPILOCK_PUMP=0`` environment variable.

MPI_THREAD_MULTIPLE requirement
-------------------------------

The pump thread calls MPI concurrently with the main thread, which requires
the MPI runtime to be initialized with ``MPI_THREAD_MULTIPLE``. ``mpi4py``
defaults to requesting it (``mpi4py.rc.thread_level = "multiple"``), but the
runtime may downgrade depending on its build and configuration. If
``MPI.Query_thread() != MPI.THREAD_MULTIPLE`` at controller construction, the
pump is not started and a warning is emitted. The lock still functions, but a
busy master will stall other ranks' acquisitions until it next re-enters MPI.

Operating envelope
------------------

The starvation pattern the pump guards against (acquisitions blocked because
the target is not inside MPI) only manifests where passive-target RMA depends
on the target making progress: the ``osc/pt2pt`` component everywhere, and
network RMA components such as ``osc/ucx`` or ``osc/rdma`` on multi-node jobs.
Single-node MPI with the ``osc/sm`` shared-memory component progresses without
target polling, so the stall does not appear there.
