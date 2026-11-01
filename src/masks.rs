//! Host memory for full-resolution masks, recycled across calls
//!
//! A 2560 px mask is 4-5 MB. Zeroing and faulting in fresh pages for 27 of them took a
//! helper thread about 27 ms, against about 5 ms of GPU work, and above glibc's 32 MB
//! mmap threshold every call gets fresh pages. Each [`MaskBuffer`] goes back to its
//! model's pool when dropped, and the next call writes into pages that are already
//! mapped

use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::thread::JoinHandle;

use eyre::{Result, eyre};

/// Returned buffers a model keeps for reuse. Two cover a caller that holds one result
/// while asking for the next; more would pin memory a caller has let go of
const KEPT: usize = 2;

type Shelf = Mutex<Vec<Vec<u8>>>;

/// Mask bytes in host memory: `n * height * width` bytes, 1 = foreground
///
/// Dropping the buffer hands its memory back to the model that filled it, which
/// reuses it for a later call
pub struct MaskBuffer {
    // every byte is initialized, and only the first `used` are mask bytes
    bytes: Vec<u8>,
    used: usize,
    pool: Weak<Shelf>,
}

impl MaskBuffer {
    fn new(bytes: Vec<u8>, used: usize, pool: Weak<Shelf>) -> Self {
        debug_assert!(used <= bytes.len());
        Self { bytes, used, pool }
    }

    /// Copies the mask bytes into a `Vec` the caller owns
    pub fn to_vec(&self) -> Vec<u8> {
        self.deref().to_vec()
    }
}

impl Deref for MaskBuffer {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.bytes[..self.used]
    }
}

impl DerefMut for MaskBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.bytes[..self.used]
    }
}

impl Drop for MaskBuffer {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade() {
            MaskPool(pool).give(std::mem::take(&mut self.bytes));
        }
    }
}

impl Clone for MaskBuffer {
    /// A copy of the mask bytes that frees its memory when dropped
    fn clone(&self) -> Self {
        let bytes = self.to_vec();
        Self::new(bytes, self.used, Weak::new())
    }
}

impl Default for MaskBuffer {
    fn default() -> Self {
        Self::new(Vec::new(), 0, Weak::new())
    }
}

impl fmt::Debug for MaskBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ones = self.iter().filter(|b| **b != 0).count();
        write!(f, "MaskBuffer({} bytes, {ones} foreground)", self.used)
    }
}

/// The buffers a model's dropped results gave back
#[derive(Default)]
pub(crate) struct MaskPool(Arc<Shelf>);

impl MaskPool {
    fn shelf(&self) -> std::sync::MutexGuard<'_, Vec<Vec<u8>>> {
        // a panic mid-push leaves a whole Vec or none, so the list stays usable
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn give(&self, bytes: Vec<u8>) {
        let mut shelf = self.shelf();
        shelf.push(bytes);
        if shelf.len() > KEPT {
            let smallest = (0..shelf.len()).min_by_key(|&i| shelf[i].len()).expect("shelf is not empty");
            shelf.swap_remove(smallest);
        }
    }

    /// Starts getting at least `len` bytes: the smallest returned buffer that is big
    /// enough, or else new memory faulted in on another thread while the GPU works
    pub(crate) fn reserve(&self, len: usize) -> Reserved {
        let mut shelf = self.shelf();
        let fit = (0..shelf.len()).filter(|&i| shelf[i].len() >= len).min_by_key(|&i| shelf[i].len());
        match fit {
            Some(i) => Reserved::Ready(shelf.swap_remove(i)),
            None => Reserved::Faulting(prefault(len)),
        }
    }

    /// The reserved memory as a buffer of `used` bytes, grown if the guess was short.
    /// Its contents are stale, so the caller overwrites all of them
    pub(crate) fn take(&self, reserved: Reserved, used: usize) -> Result<MaskBuffer> {
        let mut bytes = match reserved {
            Reserved::Ready(bytes) => bytes,
            Reserved::Faulting(thread) => thread.join().map_err(|_| eyre!("mask buffer thread panicked"))?,
        };
        if bytes.len() < used {
            // the copy into the new tail faults its pages in
            bytes.resize(used, 0);
        }
        Ok(MaskBuffer::new(bytes, used, Arc::downgrade(&self.0)))
    }

    /// Puts back memory a call reserved but didn't use
    pub(crate) fn release(&self, reserved: Reserved) {
        match reserved {
            Reserved::Ready(bytes) => self.give(bytes),
            // dropping the handle detaches the thread, which frees its buffer
            Reserved::Faulting(_) => {}
        }
    }
}

/// Memory set aside by [`MaskPool::reserve`]
pub(crate) enum Reserved {
    Ready(Vec<u8>),
    Faulting(JoinHandle<Vec<u8>>),
}

/// Makes a zeroed host buffer of `len` bytes with every page already faulted in, on
/// another thread. The driver's pageable copy into fresh pages would otherwise take the
/// faults itself, with the GPU idle
fn prefault(len: usize) -> JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut buf = vec![0u8; len];
        // black_box, or the compiler drops stores of zero into calloc'd memory
        for byte in buf.iter_mut().step_by(4096) {
            *std::hint::black_box(byte) = 0;
        }
        buf
    })
}

#[cfg(test)]
mod tests {
    use super::{KEPT, MaskPool};

    #[test]
    fn dropped_buffers_are_reused_smallest_fit_first() {
        let pool = MaskPool::default();
        let big = pool.take(pool.reserve(300), 300).unwrap();
        let small = pool.take(pool.reserve(100), 100).unwrap();
        let (big_at, small_at) = (big.as_ptr(), small.as_ptr());
        drop((big, small));

        let reused = pool.take(pool.reserve(50), 80).unwrap();
        assert_eq!((reused.as_ptr(), reused.len()), (small_at, 80));
        let reused = pool.take(pool.reserve(200), 200).unwrap();
        assert_eq!(reused.as_ptr(), big_at);
    }

    #[test]
    fn short_guess_grows() {
        let pool = MaskPool::default();
        assert_eq!(pool.take(pool.reserve(10), 40).unwrap().len(), 40);
    }

    #[test]
    fn shelf_keeps_the_largest() {
        let pool = MaskPool::default();
        let sizes = [10, 40, 20, 30];
        let buffers: Vec<_> = sizes.iter().map(|&n| pool.take(pool.reserve(n), n).unwrap()).collect();
        drop(buffers);
        let mut kept: Vec<usize> = pool.shelf().iter().map(Vec::len).collect();
        kept.sort();
        assert_eq!(KEPT, 2);
        assert_eq!(kept, [30, 40]);
    }
}
