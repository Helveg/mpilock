from mpisync import sync
import time

controller = sync()

# with controller.write():
#     print(controller._rank, "is writing")
#     time.sleep(controller._rank)
#
# import mpi4py.MPI
# mpi4py.MPI.COMM_WORLD.Barrier()
#
# with controller.read():
#     print(controller._rank, "started reading")
#     time.sleep(2)

with controller.single_write() as fence:
    fence.guard()
    print("Rank", controller._rank, "survived")

writer = 16

with controller.single_write(handle=writer) as handle:
    if handle is not None:
        print("Rank", controller._rank, "survived")
