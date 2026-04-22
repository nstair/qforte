#include "qforte_globals.h"

namespace qforte_globals {

static bool gpu_only_mode_ = false;

void set_gpu_only(bool val) {
    gpu_only_mode_ = val;
}

bool get_gpu_only() {
    return gpu_only_mode_;
}

} // namespace qforte_globals
