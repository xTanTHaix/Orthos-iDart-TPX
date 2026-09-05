use crate::qat::QatOffload;

pub enum CompressionPath<'a> {
    Cpu,
    Offload(&'a QatOffload),
}

pub fn select_offload_or_cpu<'a>(
    _sample: &[u8],
    qat: Option<&'a QatOffload>,
) -> CompressionPath<'a> {
    if let Some(dev) = qat {
        CompressionPath::Offload(dev)
    } else {
        CompressionPath::Cpu
    }
}
