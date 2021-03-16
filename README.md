[![codecov](https://codecov.io/gh/Helveg/mpilock/branch/main/graph/badge.svg?token=WQ1U6UNPGA)](https://codecov.io/gh/Helveg/mpilock)

# About

`mpilock` offers a `WindowController` class with a high-level API for parallel access to
resources. The `WindowController` can be used to perform `read`, `write` or `single_write`.

Read operations happen in parallel while write operations will lock the resource and
prevent any new read or write operations and will wait for all existing read operations to
finish. After the write operation completes the lock is released and other operations can
resume.

The `WindowController` does not contain any logic to control the resources, it only locks
and synchronizes the MPI processes. Once the operation permission is obtained it's up to
the user to perform the reading/writing to the resources.

The `sync` method is a factory for `WindowController`s and can simplify creation of
`WindowController`s.

# Example usage

```
from mpilock import sync
from h5py import File

# Create a default WindowController on `COMM_WORLD` with the master on rank 0
lock = sync()

# Fencing is the preferred idiom to fence anyone that isn't writing out of
# the writer's code block, and afterwards share a resource
with lock.single_write() as fence:
  # Makes anyone without access long jump to the end of the with statement
  fence.guard()
  resource = h5py.File("hello.world", "w")
  # Put a resource to be collected by other processes
  fence.share(resource)
resource = fence.collect()

try:
  # Acquire a parallel read lock, guarantees noone writes while you're reading.
  with lock.read():
    data = resource["/my_data"][()]
  # Acquire a write lock, will block all reading and writing.
  with lock.write():
    resource.create_dataset(lock.rank, data=data)
finally:
  with lock.single_write() as fence:
    fence.guard()
    resource.close()
```
