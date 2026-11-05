//! NVTX ranges, so nsys timelines split by pipeline stage. Without the `nvtx` feature, and
//! on macOS, where there is no CUDA, they compile to nothing

use std::ffi::CStr;

#[cfg(all(feature = "nvtx", not(target_os = "macos")))]
unsafe extern "C" {
    fn sam3rs_nvtx_push(name: *const std::ffi::c_char);
    fn sam3rs_nvtx_pop();
}

/// Open range; it closes when dropped
pub(crate) struct Range(());

/// Opens a range named `name` on the calling thread
pub(crate) fn range(name: &CStr) -> Range {
    #[cfg(all(feature = "nvtx", not(target_os = "macos")))]
    unsafe {
        sam3rs_nvtx_push(name.as_ptr())
    };
    #[cfg(not(all(feature = "nvtx", not(target_os = "macos"))))]
    let _ = name;
    Range(())
}

#[cfg(all(feature = "nvtx", not(target_os = "macos")))]
impl Drop for Range {
    fn drop(&mut self) {
        unsafe { sam3rs_nvtx_pop() };
    }
}
