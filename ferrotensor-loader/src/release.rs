//! Dropping a tensor's pages from the process once its bytes are read.
//!
//! A memory-mapped checkpoint's pages stay resident after they are read, so
//! loading every tensor would otherwise keep the whole file in the
//! process's memory until the mapping is dropped. After a tensor is read,
//! [`release`] tells the kernel the process no longer needs the pages that
//! hold only that tensor's bytes (`madvise(MADV_DONTNEED)`). The mapping is
//! read-only and backed by the file, so a later read of the same bytes
//! faults them in again from the file with the same contents.

use std::ops::Range;

/// Granularity of the released ranges: 64 KiB, a multiple of every page
/// size in common use (4, 16 and 64 KiB), so a range rounded inwards to it
/// starts and ends on page boundaries without reading the page size from
/// the system. Up to 64 KiB at each end of a tensor stays resident.
#[cfg_attr(not(unix), allow(dead_code))]
pub(crate) const RELEASE_ALIGN: usize = 64 * 1024;

/// The largest sub-range of `start..end` whose ends are multiples of
/// `align`: the start rounded up, the end rounded down. Empty (`start..start`
/// or similar) when no whole `align`-sized block fits, so pages shared with
/// a neighbouring tensor are never included.
#[cfg_attr(not(unix), allow(dead_code))]
pub(crate) fn inner_aligned(start: usize, end: usize, align: usize) -> Range<usize> {
    debug_assert!(align.is_power_of_two());
    let Some(lo) = start.checked_next_multiple_of(align) else {
        return start..start;
    };
    let hi = end - end % align;
    if lo < hi {
        lo..hi
    } else {
        lo..lo
    }
}

/// Advises the kernel that the whole pages of `mmap` covered by `data` (a
/// sub-slice of it whose borrow has ended) are no longer needed. Best
/// effort: errors are ignored, and it does nothing outside Unix.
pub(crate) fn release(mmap: &memmap2::Mmap, data: &[u8]) {
    #[cfg(unix)]
    {
        let base = mmap.as_ptr() as usize;
        let Some(start) = (data.as_ptr() as usize).checked_sub(base) else {
            return;
        };
        let end = start + data.len();
        if end > mmap.len() {
            return;
        }
        // The mapping starts on a page boundary, so offsets that are
        // multiples of RELEASE_ALIGN are page-aligned addresses too.
        let range = inner_aligned(start, end, RELEASE_ALIGN);
        if range.is_empty() {
            return;
        }
        // SAFETY: the mapping is a read-only, shared mapping of a file
        // (`MmapOptions::map`), and no reference into `range` outlives this
        // call (the caller's borrow of `data` has ended). MADV_DONTNEED
        // only drops the process's page-table entries for it; the next
        // access faults the pages in again from the file, unchanged.
        let _ = unsafe {
            mmap.unchecked_advise_range(
                memmap2::UncheckedAdvice::DontNeed,
                range.start,
                range.len(),
            )
        };
    }
    #[cfg(not(unix))]
    {
        let _ = (mmap, data);
    }
}

#[cfg(test)]
mod tests {
    use super::{inner_aligned, RELEASE_ALIGN};

    const A: usize = RELEASE_ALIGN;

    #[test]
    fn ranges_inside_one_block_are_empty() {
        assert!(inner_aligned(0, 100, A).is_empty());
        assert!(inner_aligned(10, A - 1, A).is_empty());
        assert!(inner_aligned(A + 1, 2 * A - 1, A).is_empty());
        // Crossing a boundary without covering a whole block.
        assert!(inner_aligned(A - 10, A + 10, A).is_empty());
        assert!(inner_aligned(5, 5, A).is_empty());
    }

    #[test]
    fn ranges_are_rounded_inwards() {
        assert_eq!(inner_aligned(0, A, A), 0..A);
        assert_eq!(inner_aligned(1, 3 * A + 5, A), A..3 * A);
        assert_eq!(inner_aligned(A, 3 * A, A), A..3 * A);
        assert_eq!(inner_aligned(A - 1, 2 * A, A), A..2 * A);
    }

    #[test]
    fn end_near_the_top_of_the_address_space_does_not_overflow() {
        assert!(inner_aligned(usize::MAX - 3, usize::MAX, A).is_empty());
    }

    #[test]
    fn adjacent_tensors_never_share_a_released_block() {
        // Tensors laid end to end with sizes from tiny to several blocks.
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut offset = 37; // an unaligned header, as in safetensors
        let mut released: Vec<(usize, usize, std::ops::Range<usize>)> = Vec::new();
        for _ in 0..2000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let len = (seed % (5 * A as u64)) as usize;
            let range = inner_aligned(offset, offset + len, A);
            // Inside its tensor, aligned at both ends.
            if !range.is_empty() {
                assert!(range.start >= offset && range.end <= offset + len);
                assert!(range.start.is_multiple_of(A));
                assert!(range.end.is_multiple_of(A));
            }
            released.push((offset, offset + len, range));
            offset += len;
        }
        for pair in released.windows(2) {
            let (_, _, a) = &pair[0];
            let (b_start, _, b) = &pair[1];
            if !a.is_empty() {
                assert!(a.end <= *b_start, "{a:?} reaches into the next tensor");
            }
            if !a.is_empty() && !b.is_empty() {
                assert!(a.end <= b.start, "{a:?} overlaps {b:?}");
            }
        }
    }
}
