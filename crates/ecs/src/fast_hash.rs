//! High-speed, non-cryptographic integer hasher for ECS internal lookup tables.

use std::hash::{BuildHasherDefault, Hasher};

const PRIME: u64 = 0x517c_c1b7_2722_0a95;

/// Fast multiplier-based non-cryptographic hasher.
pub struct FastHasher(u64);

impl Default for FastHasher {
    #[inline(always)]
    fn default() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
}

impl Hasher for FastHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ (b as u64)).wrapping_mul(PRIME);
        }
    }

    #[inline(always)]
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0 ^ (i as u64)).wrapping_mul(PRIME);
    }

    #[inline(always)]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0 ^ i).wrapping_mul(PRIME);
    }

    #[inline(always)]
    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0 ^ (i as u64)).wrapping_mul(PRIME);
    }
}

/// A [`std::collections::HashMap`] backed by [`FastHasher`] for high-performance internal key lookups.
pub type FastHashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FastHasher>>;
