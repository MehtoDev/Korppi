//! GPU queries and indirect commands (finding RHI-05 / audit missing-functionality matrix).
//!
//! **Timestamps**: a query set of `QueryType::Timestamp` holds `count` timestamps.
//! `write_timestamp` records the GPU time once all earlier commands have completed;
//! `resolve_query_set` (outside render passes) copies a range of them into a buffer with
//! `COPY_DST` usage as `u64` ticks and makes them writable again. Each query is written once
//! between resolves. Multiply tick differences by `DeviceLimits::timestamp_period_ns` for
//! nanoseconds; make the buffer visible to the host with a barrier from
//! `TRANSFER`/`TRANSFER_WRITE` to `HOST`/`HOST_READ`. Needs `DeviceFeatures::TIMESTAMP_QUERY`.
//!
//! **Indirect commands** read their arguments from a buffer with `INDIRECT` usage, in the
//! layouts below (the same in Vulkan and OpenGL). Record writes to that buffer and then a
//! barrier to `DRAW_INDIRECT`/`INDIRECT_COMMAND_READ`. Several draws per call need
//! `DeviceFeatures::MULTI_DRAW_INDIRECT`, a non-zero `first_instance`
//! `DeviceFeatures::INDIRECT_FIRST_INSTANCE`. Indexed indirect draws use the index buffer bound
//! at offset 0 (OpenGL cannot offset it).

/// Kind of query in a query set.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum QueryType {
    /// GPU timestamps.
    Timestamp,
}

/// Parameters for creating a query set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuerySetDescriptor {
    /// Kind of queries.
    pub ty: QueryType,
    /// Number of queries (at least 1).
    pub count: u32,
    /// Optional debug label.
    pub label: Option<&'static str>,
}

/// Arguments of one `draw_indirect` draw (16 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DrawIndirectArgs {
    /// Vertices per instance.
    pub vertex_count: u32,
    /// Instances.
    pub instance_count: u32,
    /// First vertex.
    pub first_vertex: u32,
    /// First instance.
    pub first_instance: u32,
}

/// Arguments of one `draw_indexed_indirect` draw (20 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DrawIndexedIndirectArgs {
    /// Indices per instance.
    pub index_count: u32,
    /// Instances.
    pub instance_count: u32,
    /// First index.
    pub first_index: u32,
    /// Value added to each index.
    pub base_vertex: i32,
    /// First instance.
    pub first_instance: u32,
}

/// Arguments of one `dispatch_indirect` dispatch (12 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DispatchIndirectArgs {
    /// Workgroups in x.
    pub x: u32,
    /// Workgroups in y.
    pub y: u32,
    /// Workgroups in z.
    pub z: u32,
}

macro_rules! as_bytes {
    ($ty:ty) => {
        impl $ty {
            /// Size of one record in bytes.
            pub const SIZE: u64 = std::mem::size_of::<Self>() as u64;

            /// The record's bytes, as the GPU reads them.
            pub fn to_bytes(&self) -> Vec<u8> {
                // SAFETY: the type is `repr(C)` with only 4-byte integer fields, so it has no
                // padding and every byte is initialized.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        (self as *const Self).cast::<u8>(),
                        std::mem::size_of::<Self>(),
                    )
                };
                bytes.to_vec()
            }
        }
    };
}
as_bytes!(DrawIndirectArgs);
as_bytes!(DrawIndexedIndirectArgs);
as_bytes!(DispatchIndirectArgs);

/// Checks an indirect draw's argument range: 4-byte aligned offset, stride a multiple of 4 and
/// at least `record_size` (when several records are read), and the last record inside the
/// buffer (if its size is known).
pub fn check_indirect_range(
    offset: u64,
    draw_count: u32,
    stride: u32,
    record_size: u64,
    buffer_size: Option<u64>,
) -> Result<(), String> {
    if !offset.is_multiple_of(4) {
        return Err(format!("offset {offset} is not a multiple of 4"));
    }
    if draw_count > 1 && (!stride.is_multiple_of(4) || u64::from(stride) < record_size) {
        return Err(format!(
            "stride {stride} must be a multiple of 4 of at least {record_size} bytes"
        ));
    }
    let count_sub_1 = u64::from(draw_count.saturating_sub(1));
    let stride_u64 = u64::from(stride);
    let span = count_sub_1.checked_mul(stride_u64);
    let end = span
        .and_then(|s| offset.checked_add(s))
        .and_then(|o| o.checked_add(record_size));
    match buffer_size {
        Some(size) if draw_count > 0 => match end {
            Some(end) if end > size => Err(format!(
                "the arguments end at byte {end} of a {size}-byte buffer"
            )),
            None => Err(format!("the arguments end outside the {size}-byte buffer")),
            _ => Ok(()),
        },
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_have_the_api_layouts() {
        assert_eq!(DrawIndirectArgs::SIZE, 16);
        assert_eq!(DrawIndexedIndirectArgs::SIZE, 20);
        assert_eq!(DispatchIndirectArgs::SIZE, 12);
        let args = DrawIndexedIndirectArgs {
            index_count: 1,
            instance_count: 2,
            first_index: 3,
            base_vertex: -1,
            first_instance: 5,
        };
        let bytes = args.to_bytes();
        assert_eq!(&bytes[12..16], &(-1_i32).to_ne_bytes());
        assert_eq!(&bytes[16..20], &5_u32.to_ne_bytes());
    }

    #[test]
    fn indirect_ranges_are_checked() {
        assert!(check_indirect_range(0, 2, 16, 16, Some(32)).is_ok());
        assert!(check_indirect_range(0, 3, 16, 16, Some(32)).is_err());
        assert!(check_indirect_range(2, 1, 16, 16, None).is_err());
        assert!(check_indirect_range(0, 2, 12, 16, None).is_err());
        assert!(check_indirect_range(0, 1, 0, 20, Some(20)).is_ok());
    }
}
