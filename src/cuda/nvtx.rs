//! NVTX ranges, so nsys timelines split by pipeline stage. Without the `nvtx` feature
//! they compile to nothing

use std::ffi::CStr;

#[cfg(feature = "nvtx")]
unsafe extern "C" {
    fn sam3rs_nvtx_push(name: *const std::ffi::c_char);
    fn sam3rs_nvtx_pop();
}

/// Open range; it closes when dropped
pub struct Range(());

/// Opens a range named `name` on the calling thread
pub fn range(name: &CStr) -> Range {
    #[cfg(feature = "nvtx")]
    unsafe {
        sam3rs_nvtx_push(name.as_ptr())
    };
    #[cfg(not(feature = "nvtx"))]
    let _ = name;
    Range(())
}

#[cfg(feature = "nvtx")]
impl Drop for Range {
    fn drop(&mut self) {
        unsafe { sam3rs_nvtx_pop() };
    }
}
