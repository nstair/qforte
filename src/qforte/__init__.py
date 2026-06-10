__version__ = '0.1'
__author__ = 'Qforte Dev'
#sys.path.insert(1, os.path.abspath('.'))

from .qforte import *
from qforte.abc import *
from qforte.adapters import *
from qforte.helper import *
from qforte.maths import *
from qforte.experiment import *
from qforte.ite import *
from qforte.qkd import *
from qforte.qpea import *
from qforte.system import *
from qforte.hva import *
from qforte.ucc import *
from qforte.utils import *
from qforte.fqe_api import *
from qforte.cusv_api import *

def gpu_only(val=None):
    """Get or set the global gpu_only flag.
    
    When set to True, newly created TensorGPU objects will skip host-side
    memory allocation, reducing memory usage for GPU-only operations.
    
    Args:
        val: If provided (bool), sets the flag and returns None.
             If omitted, returns the current flag value.
    
    Example:
        qforte.gpu_only(True)   # enable gpu_only mode
        qforte.gpu_only(False)  # disable gpu_only mode
        qforte.gpu_only()       # returns current value
    """
    try:
        from .qforte import set_gpu_only, get_gpu_only
        if val is None:
            return get_gpu_only()
        set_gpu_only(val)
    except ImportError:
        if val is None:
            return False
        if val:
            raise RuntimeError("gpu_only mode requires CUDA-enabled qforte build.")
