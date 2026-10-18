//! Hybrid acoustic synthesis combining tapped delay lines, Feedback Delay Networks, and triple buffering.

pub mod fdn;
pub mod tapped_delay;
pub mod triple_buffer;

pub use fdn::{householder_diffuse_8, FdnConfig, FeedbackDelayNetwork};
pub use tapped_delay::{EarlyReflectionTap, TappedDelayLine};
pub use triple_buffer::{
    triple_buffer, triple_buffer_with, TripleBufferConsumer, TripleBufferProducer,
};
