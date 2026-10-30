// NVTX3 is header-only, so Rust reaches it through these two exported functions. Outside
// a profiler the calls resolve to no-ops after the first one
#include <nvtx3/nvToolsExt.h>

void sam3rs_nvtx_push(const char* name) { nvtxRangePushA(name); }
void sam3rs_nvtx_pop(void) { nvtxRangePop(); }
