//! Size flags controlling container space distribution and alignment.

use korppi_core::bitflags;

bitflags! {
    /// Size flags controlling how container layouts partition space among child controls.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
    pub struct SizeFlags: u32 {
        /// Do not expand or fill; align to start of allocated slot.
        const SHRINK_BEGIN = 0;
        /// Fill the entire allocated slot along the axis.
        const FILL = 1;
        /// Request extra expansion space from parent container.
        const EXPAND = 2;
        /// Center inside allocated slot along the axis.
        const SHRINK_CENTER = 4;
        /// Align to end of allocated slot along the axis.
        const SHRINK_END = 8;
        /// Combination of EXPAND and FILL.
        const EXPAND_FILL = Self::EXPAND.bits() | Self::FILL.bits();
    }
}
