#ifndef _qforte_globals_h_
#define _qforte_globals_h_

/// Global configuration flags for qforte.
namespace qforte_globals {

/// When true, TensorGPU constructors skip host-side memory allocation.
/// This is useful for GPU-only subroutines (e.g. inside FCIComputerGPU)
/// where host memory would never be used.
void set_gpu_only(bool val);
bool get_gpu_only();

} // namespace qforte_globals

#endif // _qforte_globals_h_
