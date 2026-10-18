//! Zero-dependency bitflags macro for type-safe bitmask flag sets (`Rule 2`, `FIX-11.2`).
//!
//! Generates a `#[repr(transparent)]` wrapper around an unsigned integer primitive (`u8`, `u16`,
//! `u32`, `u64`, `u128`, `usize`) with `const fn` set operations and standard bitwise operators.

/// Internal helper macro for [`bitflags!`] that strips `Debug` from `#[derive(...)]` so
/// [`bitflags!`] can provide a human-readable symbolic `Debug` implementation (`Name(FLAG_A | FLAG_B)`).
#[doc(hidden)]
#[macro_export]
macro_rules! __bitflags_struct {
    (
        @filter_derive
        kept: [$($kept:ident),*]
        rest: []
        attrs_in: [$($attrs_in:tt)*]
        attrs_out: [$($attrs_out:tt)*]
        vis: [$vis:vis]
        name: $Name:ident
        ty: $T:ty
    ) => {
        $crate::__bitflags_struct! {
            attrs_in: [$($attrs_in)*]
            attrs_out: [$($attrs_out)* [derive($($kept),*)]]
            vis: [$vis]
            name: $Name
            ty: $T
        }
    };
    (
        @filter_derive
        kept: [$($kept:ident),*]
        rest: [Debug $(, $rest:ident)* $(,)?]
        attrs_in: [$($attrs_in:tt)*]
        attrs_out: [$($attrs_out:tt)*]
        vis: [$vis:vis]
        name: $Name:ident
        ty: $T:ty
    ) => {
        $crate::__bitflags_struct! {
            @filter_derive
            kept: [$($kept),*]
            rest: [$($rest),*]
            attrs_in: [$($attrs_in)*]
            attrs_out: [$($attrs_out)*]
            vis: [$vis]
            name: $Name
            ty: $T
        }
    };
    (
        @filter_derive
        kept: [$($kept:ident),*]
        rest: [$first:ident $(, $rest:ident)* $(,)?]
        attrs_in: [$($attrs_in:tt)*]
        attrs_out: [$($attrs_out:tt)*]
        vis: [$vis:vis]
        name: $Name:ident
        ty: $T:ty
    ) => {
        $crate::__bitflags_struct! {
            @filter_derive
            kept: [$($kept,)* $first]
            rest: [$($rest),*]
            attrs_in: [$($attrs_in)*]
            attrs_out: [$($attrs_out)*]
            vis: [$vis]
            name: $Name
            ty: $T
        }
    };
    (
        attrs_in: [[derive($($derives:tt)*)] $($rest_attrs:tt)*]
        attrs_out: [$($attrs_out:tt)*]
        vis: [$vis:vis]
        name: $Name:ident
        ty: $T:ty
    ) => {
        $crate::__bitflags_struct! {
            @filter_derive
            kept: []
            rest: [$($derives)*]
            attrs_in: [$($rest_attrs)*]
            attrs_out: [$($attrs_out)*]
            vis: [$vis]
            name: $Name
            ty: $T
        }
    };
    (
        attrs_in: [[$($other:tt)+] $($rest_attrs:tt)*]
        attrs_out: [$($attrs_out:tt)*]
        vis: [$vis:vis]
        name: $Name:ident
        ty: $T:ty
    ) => {
        $crate::__bitflags_struct! {
            attrs_in: [$($rest_attrs)*]
            attrs_out: [$($attrs_out)* [$($other)+]]
            vis: [$vis]
            name: $Name
            ty: $T
        }
    };
    (
        attrs_in: []
        attrs_out: [$([$($attrs_out:tt)+])*]
        vis: [$vis:vis]
        name: $Name:ident
        ty: $T:ty
    ) => {
        $(#[$($attrs_out)+])*
        #[repr(transparent)]
        $vis struct $Name($T);
    };
}

/// Declares one or more strongly-typed bitflag structs backed by an integer primitive.
///
/// # Inputs / Outputs
/// Accepts one or more `struct Name: <int_type> { const FLAG = <expr>; ... }` definitions and
/// expands to `#[repr(transparent)]` structs implementing `empty`, `all`, `bits`, `from_bits`,
/// `from_bits_truncate`, `from_bits_retain`, `is_empty`, `is_all`, `intersects`, `contains`,
/// `insert`, `remove`, `toggle`, `set`, `union`, `intersection`, `difference`,
/// `symmetric_difference`, `complement`, symbolic `Debug`, and standard bitwise operator traits.
#[macro_export]
macro_rules! bitflags {
    (
        $(
            $(#[$($outer:tt)+])*
            $vis:vis struct $Name:ident: $T:ty {
                $(
                    $(#[$inner:meta])*
                    const $FLAG:ident = $value:expr;
                )*
            }
        )+
    ) => {
        $(
            $crate::__bitflags_struct! {
                attrs_in: [$([$($outer)+])*]
                attrs_out: []
                vis: [$vis]
                name: $Name
                ty: $T
            }

            impl ::core::fmt::Debug for $Name {
                fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                    write!(f, "{}(", stringify!($Name))?;
                    let mut remaining = self.0;
                    let mut first = true;
                    $(
                        {
                            let flag_bits: $T = $value;
                            if flag_bits != 0 && (self.0 & flag_bits) == flag_bits {
                                if !first {
                                    write!(f, " | ")?;
                                }
                                first = false;
                                write!(f, "{}", stringify!($FLAG))?;
                                remaining &= !flag_bits;
                            }
                        }
                    )*
                    if remaining != 0 {
                        if !first {
                            write!(f, " | ")?;
                        }
                        write!(f, "{remaining:#x}")?;
                    } else if first {
                        write!(f, "0x0")?;
                    }
                    write!(f, ")")
                }
            }

            impl $Name {
                $(
                    $(#[$inner])*
                    pub const $FLAG: Self = Self($value);
                )*

                /// Returns an empty set of flags with no bits set (`0`).
                #[inline]
                pub const fn empty() -> Self {
                    Self(0)
                }

                /// Returns the set of all defined flags OR-ed together.
                #[inline]
                pub const fn all() -> Self {
                    Self(0 $(| ($value))*)
                }

                /// Returns the underlying raw integer bits.
                #[inline]
                pub const fn bits(&self) -> $T {
                    self.0
                }

                /// Converts from raw bits if and only if all set bits correspond to defined flags.
                #[inline]
                pub const fn from_bits(bits: $T) -> Option<Self> {
                    if (bits & !Self::all().bits()) == 0 {
                        Some(Self(bits))
                    } else {
                        None
                    }
                }

                /// Converts from raw bits, truncating any unknown bits not in [`Self::all`].
                #[inline]
                pub const fn from_bits_truncate(bits: $T) -> Self {
                    Self(bits & Self::all().bits())
                }

                /// Converts from raw bits, retaining all bits verbatim (including unknown bits).
                #[inline]
                pub const fn from_bits_retain(bits: $T) -> Self {
                    Self(bits)
                }

                /// Returns `true` if no flags are set.
                #[inline]
                pub const fn is_empty(&self) -> bool {
                    self.0 == 0
                }

                /// Returns `true` if all defined flags in [`Self::all`] are set.
                #[inline]
                pub const fn is_all(&self) -> bool {
                    let all = Self::all().0;
                    (self.0 & all) == all
                }

                /// Returns `true` if `self` and `other` share at least one set flag.
                #[inline]
                pub const fn intersects(&self, other: Self) -> bool {
                    (self.0 & other.0) != 0
                }

                /// Returns `true` if all flags in `other` are set in `self`.
                #[inline]
                pub const fn contains(&self, other: Self) -> bool {
                    (self.0 & other.0) == other.0
                }

                /// Inserts the specified flags in-place.
                #[inline]
                pub fn insert(&mut self, other: Self) {
                    self.0 |= other.0;
                }

                /// Removes the specified flags in-place.
                #[inline]
                pub fn remove(&mut self, other: Self) {
                    self.0 &= !other.0;
                }

                /// Toggles the specified flags in-place.
                #[inline]
                pub fn toggle(&mut self, other: Self) {
                    self.0 ^= other.0;
                }

                /// Inserts or removes the specified flags according to `value`.
                #[inline]
                pub fn set(&mut self, other: Self, value: bool) {
                    if value {
                        self.insert(other);
                    } else {
                        self.remove(other);
                    }
                }

                /// Returns the bitwise union (`self | other`).
                #[inline]
                pub const fn union(self, other: Self) -> Self {
                    Self(self.0 | other.0)
                }

                /// Returns the bitwise intersection (`self & other`).
                #[inline]
                pub const fn intersection(self, other: Self) -> Self {
                    Self(self.0 & other.0)
                }

                /// Returns the bitwise difference (`self & !other`).
                #[inline]
                pub const fn difference(self, other: Self) -> Self {
                    Self(self.0 & !other.0)
                }

                /// Returns the bitwise symmetric difference (`self ^ other`).
                #[inline]
                pub const fn symmetric_difference(self, other: Self) -> Self {
                    Self(self.0 ^ other.0)
                }

                /// Returns the complement within defined flags (`!self & Self::all()`).
                #[inline]
                pub const fn complement(self) -> Self {
                    Self(!self.0 & Self::all().0)
                }
            }

            impl ::core::ops::BitOr for $Name {
                type Output = Self;
                #[inline]
                fn bitor(self, rhs: Self) -> Self {
                    Self(self.0 | rhs.0)
                }
            }

            impl ::core::ops::BitOrAssign for $Name {
                #[inline]
                fn bitor_assign(&mut self, rhs: Self) {
                    self.0 |= rhs.0;
                }
            }

            impl ::core::ops::BitAnd for $Name {
                type Output = Self;
                #[inline]
                fn bitand(self, rhs: Self) -> Self {
                    Self(self.0 & rhs.0)
                }
            }

            impl ::core::ops::BitAndAssign for $Name {
                #[inline]
                fn bitand_assign(&mut self, rhs: Self) {
                    self.0 &= rhs.0;
                }
            }

            impl ::core::ops::BitXor for $Name {
                type Output = Self;
                #[inline]
                fn bitxor(self, rhs: Self) -> Self {
                    Self(self.0 ^ rhs.0)
                }
            }

            impl ::core::ops::BitXorAssign for $Name {
                #[inline]
                fn bitxor_assign(&mut self, rhs: Self) {
                    self.0 ^= rhs.0;
                }
            }

            impl ::core::ops::Sub for $Name {
                type Output = Self;
                #[inline]
                fn sub(self, rhs: Self) -> Self {
                    Self(self.0 & !rhs.0)
                }
            }

            impl ::core::ops::SubAssign for $Name {
                #[inline]
                fn sub_assign(&mut self, rhs: Self) {
                    self.0 &= !rhs.0;
                }
            }

            impl ::core::ops::Not for $Name {
                type Output = Self;
                #[inline]
                fn not(self) -> Self {
                    self.complement()
                }
            }
        )+
    };
}
