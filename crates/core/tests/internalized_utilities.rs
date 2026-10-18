//! Primary outcome, failure-mode, and soundness tests for `FIX-11.2`:
//! Internalized `bitflags!`, `pod` (`Zeroable`, `Pod`, `bytes_of`, `cast_slice`,
//! `pod_read_unaligned`, `impl_pod!`), and `SmallVec` / `smallvec!` in `korppi_core`,
//! and verification that `thiserror`, `bitflags`, `bytemuck`, and `smallvec` are
//! removed from all workspace manifests.

use korppi_core::pod::{
    bytes_of, bytes_of_mut, cast_slice, cast_slice_mut, pod_read_unaligned, Zeroable,
};
use korppi_core::{bitflags, impl_pod, smallvec, SmallVec};
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

bitflags! {
    /// Test 32-bit flag set exercising `korppi_core::bitflags!`.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
    pub struct TestFlags: u32 {
        /// First flag.
        const ALPHA = 1 << 0;
        /// Second flag.
        const BETA  = 1 << 1;
        /// Third flag.
        const GAMMA = 1 << 2;
        /// Combined alpha and beta.
        const ALPHA_BETA = Self::ALPHA.bits() | Self::BETA.bits();
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
struct GpuVertex {
    pos: [f32; 3],
    color: u32,
}

impl_pod!(GpuVertex = 16);

#[test]
fn bitflags_operations_and_invariants() {
    let empty = TestFlags::empty();
    assert!(empty.is_empty());
    assert!(!empty.is_all());
    assert_eq!(empty.bits(), 0);
    assert_eq!(TestFlags::default(), empty);

    let all = TestFlags::all();
    assert!(all.is_all());
    assert_eq!(all.bits(), 0b111);

    let mut flags = TestFlags::ALPHA;
    assert!(flags.contains(TestFlags::ALPHA));
    assert!(!flags.contains(TestFlags::BETA));
    assert!(flags.intersects(TestFlags::ALPHA_BETA));

    flags.insert(TestFlags::BETA);
    assert_eq!(flags, TestFlags::ALPHA_BETA);
    assert!(flags.contains(TestFlags::ALPHA | TestFlags::BETA));

    flags.toggle(TestFlags::ALPHA | TestFlags::GAMMA);
    assert_eq!(flags, TestFlags::BETA | TestFlags::GAMMA);

    flags.remove(TestFlags::BETA);
    assert_eq!(flags, TestFlags::GAMMA);

    flags.set(TestFlags::ALPHA, true);
    flags.set(TestFlags::GAMMA, false);
    assert_eq!(flags, TestFlags::ALPHA);

    assert_eq!(
        TestFlags::from_bits(0b101),
        Some(TestFlags::ALPHA | TestFlags::GAMMA)
    );
    assert_eq!(TestFlags::from_bits(0b1000), None);
    assert_eq!(
        TestFlags::from_bits_truncate(0b1101),
        TestFlags::ALPHA | TestFlags::GAMMA
    );
    assert_eq!(TestFlags::from_bits_retain(0b1101).bits(), 0b1101);

    let a = TestFlags::ALPHA | TestFlags::BETA;
    let b = TestFlags::BETA | TestFlags::GAMMA;
    assert_eq!(a | b, all);
    assert_eq!(a & b, TestFlags::BETA);
    assert_eq!(a ^ b, TestFlags::ALPHA | TestFlags::GAMMA);
    assert_eq!(a - b, TestFlags::ALPHA);
    assert_eq!(!a, TestFlags::GAMMA);
    assert_eq!(a.union(b), all);
    assert_eq!(a.intersection(b), TestFlags::BETA);
    assert_eq!(a.difference(b), TestFlags::ALPHA);
    assert_eq!(
        a.symmetric_difference(b),
        TestFlags::ALPHA | TestFlags::GAMMA
    );
    assert_eq!(a.complement(), TestFlags::GAMMA);
}

#[test]
fn pod_zeroable_and_byte_casts_round_trip() {
    let mut zero_vtx = GpuVertex::zeroed();
    assert_eq!(
        zero_vtx,
        GpuVertex {
            pos: [0.0, 0.0, 0.0],
            color: 0,
        }
    );

    let vtx = GpuVertex {
        pos: [1.0, -2.5, 4.0],
        color: 0xDEAD_BEEF,
    };
    let raw = bytes_of(&vtx);
    assert_eq!(raw.len(), 16);

    bytes_of_mut(&mut zero_vtx).copy_from_slice(raw);
    assert_eq!(zero_vtx, vtx);

    // Unaligned read from an offset buffer.
    let mut padded = [0u8; 19];
    padded[3..19].copy_from_slice(raw);
    let recovered: GpuVertex = pod_read_unaligned(&padded[3..19]);
    assert_eq!(recovered, vtx);

    // Slice casts (both immutable and mutable).
    let vertices = [
        vtx,
        GpuVertex {
            pos: [0.5, 1.5, 2.5],
            color: 0x1122_3344,
        },
    ];
    let byte_slice: &[u8] = cast_slice(&vertices);
    assert_eq!(byte_slice.len(), 32);
    let u32_slice: &[u32] = cast_slice(&vertices);
    assert_eq!(u32_slice.len(), 8);
    assert_eq!(u32_slice[3], 0xDEAD_BEEF);
    assert_eq!(u32_slice[7], 0x1122_3344);

    let mut words = [0u32; 4];
    cast_slice_mut::<u32, u8>(&mut words).copy_from_slice(raw);
    let back_vtx: &[GpuVertex] = cast_slice(&words);
    assert_eq!(back_vtx[0], vtx);
}

#[test]
#[should_panic(expected = "cast_slice")]
fn pod_cast_slice_panics_on_size_mismatch() {
    let bytes = [1u8, 2, 3, 4, 5];
    let _: &[u32] = cast_slice(&bytes);
}

#[test]
#[should_panic(expected = "cast_slice")]
fn pod_cast_slice_panics_on_misaligned_slice() {
    let aligned_words = [0u32; 4];
    let bytes: &[u8] = cast_slice(&aligned_words);
    // Offset by 1 byte so alignment for `u32` is violated while length (12) is still a multiple of 4.
    let _: &[u32] = cast_slice(&bytes[1..13]);
}

#[test]
#[should_panic(expected = "pod_read_unaligned")]
fn pod_read_unaligned_panics_on_wrong_byte_length() {
    let bytes = [0u8; 15];
    let _: GpuVertex = pod_read_unaligned(&bytes);
}

#[test]
fn smallvec_inline_and_spilled_lifecycle_and_drop_soundness() {
    #[derive(Clone, Debug)]
    struct DropTracker {
        id: usize,
        drops: Arc<AtomicUsize>,
    }
    impl PartialEq for DropTracker {
        fn eq(&self, other: &Self) -> bool {
            self.id == other.id
        }
    }
    impl Drop for DropTracker {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    let drops = Arc::new(AtomicUsize::new(0));
    {
        let mut sv: SmallVec<[DropTracker; 4]> = SmallVec::new();
        assert!(sv.is_empty());
        assert_eq!(sv.capacity(), 4);
        assert!(!sv.spilled());

        for i in 0..4 {
            sv.push(DropTracker {
                id: i,
                drops: Arc::clone(&drops),
            });
        }
        assert_eq!(sv.len(), 4);
        assert!(!sv.spilled());

        // Pushing a 5th element spills to the heap.
        sv.push(DropTracker {
            id: 4,
            drops: Arc::clone(&drops),
        });
        assert_eq!(sv.len(), 5);
        assert!(sv.spilled());
        assert!(sv.capacity() >= 8);
        assert_eq!(
            sv.iter().map(|d| d.id).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );

        // Insert and remove preserve order and drop removed elements once.
        sv.insert(
            2,
            DropTracker {
                id: 99,
                drops: Arc::clone(&drops),
            },
        );
        assert_eq!(
            sv.iter().map(|d| d.id).collect::<Vec<_>>(),
            vec![0, 1, 99, 2, 3, 4]
        );

        let removed = sv.remove(2);
        assert_eq!(removed.id, 99);
        drop(removed);
        assert_eq!(drops.load(Ordering::SeqCst), 1);

        let cloned = sv.clone();
        assert_eq!(cloned, sv);
        drop(cloned);
        assert_eq!(drops.load(Ordering::SeqCst), 1 + 5);

        let mut drained_ids = Vec::new();
        for item in sv {
            drained_ids.push(item.id);
        }
        assert_eq!(drained_ids, vec![0, 1, 2, 3, 4]);
    }
    // 1 (removed) + 5 (cloned) + 5 (into_iter items dropped with drained loop) = 11.
    assert_eq!(drops.load(Ordering::SeqCst), 11);

    // Macro forms and slice operations.
    let mut nums: SmallVec<[i32; 4]> = smallvec![3, 1, 2, 2, 4];
    assert!(nums.spilled());
    nums.sort_unstable();
    nums.dedup();
    assert_eq!(nums.as_slice(), &[1, 2, 3, 4]);

    let repeated: SmallVec<[u8; 4]> = smallvec![7u8; 3];
    assert!(!repeated.spilled());
    assert_eq!(repeated.as_slice(), &[7, 7, 7]);
}

#[test]
fn workspace_has_removed_thiserror_bitflags_bytemuck_and_smallvec() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");

    let output = Command::new(env!("CARGO"))
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(root)
        .output()
        .expect("run cargo metadata");
    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).expect("utf-8");
    for forbidden in ["thiserror", "bitflags", "bytemuck", "smallvec"] {
        let pattern = format!("\"name\":\"{forbidden}\"");
        assert!(
            !stdout.contains(&pattern),
            "Forbidden utility crate `{forbidden}` is still present in workspace dependencies"
        );
    }
}
