import qforte as qf

shape = [2, 2]

my_tensorGPU = qf.TensorGPU(shape, "TensorGPU 1")
other = qf.TensorGPU(shape, "TensorGPU 2")
# my_tensorGPU.zero()

my_tensorGPU.set([0,0], 1.0)
my_tensorGPU.set([1,1], 2.0)

other.set([0,0], 1.0)
other.set([0,1], 1.0)
other.set([1,1], 1.0)


my_tensorGPU.to_gpu()
other.to_gpu()

my_tensorGPU.add2(other)

my_tensorGPU.from_gpu()

print(my_tensorGPU.read_data())


"""
undefined symbol: add_wrapper

issue is probably in the cmake file but i cant figure out how 
it works or what needs to be changed

If I add the .cu file to the pybind line it gives a "no such file or directory"
error for some random library "libpthread" and "libpthread_nonshared"

tried to do conda install pthread-stubs -c conda-forge but it still didn't work

apparently this is a common cmake bug? and that the pthread thing isn't even the real error?

i tried to delete the build files and now it infinite loops when i try to compile it
i give up im going home
"""