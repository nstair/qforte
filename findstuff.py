import os, glob
p = os.environ.get('CONDA_PREFIX') or ''
hits = glob.glob(p + '/targets/x86_64-linux/include/thrust/host_vector.h')
# hits = glob.glob('/usr/local/cuda/include/thrust/host_vector.h')
print('thrust header found:', bool(hits), '->', hits[:1])
