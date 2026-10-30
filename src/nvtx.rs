//! NVTX ranges, so nsys timelines split by pipeline stage

use std::ffi::{CStr, c_char};

unsafe extern "C" {
    fn sam3rs_nvtx_push(name: *const c_char);
    fn sam3rs_nvtx_pop();
}

/// Open range; it closes when dropped
pub struct Range(());

/// Opens a range named `name` on the calling thread
pub fn range(name: &CStr) -> Range {
    unsafe { sam3rs_nvtx_push(name.as_ptr()) };
    Range(())
}

impl Drop for Range {
    fn drop(&mut self) {
        unsafe { sam3rs_nvtx_pop() };
    }
}
