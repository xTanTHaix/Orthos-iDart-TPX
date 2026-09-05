pub mod offload_select;
pub mod qat;

pub use offload_select::{select_offload_or_cpu, CompressionPath};
pub use qat::QatOffload;
