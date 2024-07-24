import qforte as qf
import numpy as np
shape = [1000, 1000]

my_tensorGPU = qf.TensorGPU(shape, "TensorGPU 1")
my_tensorGPU2 = qf.TensorGPU(shape, "TensorGPU 2")
other = qf.Tensor(shape, "Tensor 1")
other2 = qf.Tensor(shape, "Tensor 2")
# my_tensorGPU.zero()

random_array = np.random.rand(my_tensorGPU.shape()[0], my_tensorGPU.shape()[1])
random = np.array(random_array, dtype = np.dtype(np.complex128))


my_tensorGPU.fill_from_nparray(random.ravel(), my_tensorGPU.shape())
my_tensorGPU2.fill_from_nparray(random.ravel(), my_tensorGPU2.shape())
other.fill_from_nparray(random.ravel(), other.shape())
other2.fill_from_nparray(random.ravel(), other2.shape())

# rand_nrm = my_tensorGPU.norm()
# other_nrm = other.norm()

my_tensorGPU.to_gpu()
my_tensorGPU2.to_gpu()

timer1 = qf.local_timer()
timer1.reset()

my_tensorGPU.add2(my_tensorGPU2)

timer1.record("gpu")
other.add(other2)
timer1.record("normal")
my_tensorGPU.to_cpu()
gpu_nrm = my_tensorGPU.norm()
normal_nrm = other.norm()

print(timer1)
# print(gpu_nrm - normal_nrm)

print('{0:.25f}'.format(gpu_nrm - normal_nrm))

# print(my_tensorGPU.read_data())


"""

apply_individual_nbody1_accumulate_gpu

"""