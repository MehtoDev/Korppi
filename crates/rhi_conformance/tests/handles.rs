//! Generational handles (FIX-1.5; finding RHI-10): freed slots are reused, and a handle to a
//! destroyed resource never reaches the resource that reuses its slot.

use korppi_rhi::{BufferDescriptor, BufferUsage, MemoryUsage, RhiError, SamplerDescriptor};
use korppi_rhi_conformance::{Backend, TestDevice};

#[test]
fn stale_handles_are_rejected_after_their_slot_is_reused() {
    for backend in Backend::ALL {
        let device = TestDevice::open(backend);
        let desc = BufferDescriptor {
            size: 16,
            usage: BufferUsage::COPY_SRC,
            memory: MemoryUsage::CpuToGpu,
            label: None,
        };
        let old = device.create_buffer(&desc, Some(&[1; 16])).expect("buffer");
        device.destroy_buffer(old).expect("destroy");
        let new = device.create_buffer(&desc, Some(&[2; 16])).expect("buffer");
        assert_eq!(new.id(), old.id(), "{backend:?}: the slot is reused");
        assert_ne!(new.generation(), old.generation(), "{backend:?}");

        let stale = |result: Result<(), RhiError>, what: &str| {
            assert!(
                matches!(result, Err(RhiError::InvalidHandle(_))),
                "{backend:?}: {what} with a stale handle: {result:?}"
            );
        };
        stale(device.write_buffer(old, 0, &[9; 4]), "write_buffer");
        stale(device.read_buffer(old, 0, 4).map(|_| ()), "read_buffer");
        stale(device.destroy_buffer(old), "destroy_buffer");
        assert_eq!(
            device.read_buffer(new, 0, 4).expect("read"),
            [2; 4],
            "{backend:?}: the new buffer is untouched"
        );

        let sampler = device
            .create_sampler(&SamplerDescriptor::default())
            .expect("sampler");
        device.destroy_sampler(sampler).expect("destroy");
        stale(device.destroy_sampler(sampler), "destroy_sampler");
    }
}
