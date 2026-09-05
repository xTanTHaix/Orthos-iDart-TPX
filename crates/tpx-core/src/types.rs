use std::fmt;
use thiserror::Error;

// ---------------------------------------------------------------------------
// File format constants
// ---------------------------------------------------------------------------

pub const MAGIC: [u8; 4] = *b"TPX\x01";
pub const MAGIC_END_FINAL: [u8; 4] = *b"1XPT";
pub const MAGIC_END_CHECKPOINT: [u8; 4] = *b"1XPC";

pub const HEADER_SIZE: usize = 48;
pub const FOOTER_SIZE: usize = 24;
pub const CHUNK_HEADER_SIZE: usize = 16;
pub const BLOCK_ENTRY_SIZE: usize = 4;

pub const DEFAULT_CHUNK_ALIGN: u32 = 4096;
pub const HUGEPAGE_CHUNK_ALIGN: u32 = 2 * 1024 * 1024;
pub const DEFAULT_BLOCK_SIZE: u32 = 64 * 1024; // 64 KB block size default
pub const DEFAULT_CHECKPOINT_INTERVAL_BYTES: u64 = 64 * 1024 * 1024;
pub const DEFAULT_CHECKPOINT_INTERVAL_MS: u64 = 60000;

pub const FLAG_DIRECT_IO: u16 = 0x01;
pub const FLAG_CHECKSUMMED: u16 = 0x02;
pub const FLAG_GDS_ASSISTED: u16 = 0x04;
pub const FLAG_HUGEPAGES: u16 = 0x08;
pub const FLAG_HW_OFFLOAD: u16 = 0x10;
pub const FLAG_SPDK: u16 = 0x20;

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DTypeId {
    Float64 = 0,
    Float32 = 1,
    Float16 = 2,
    BFloat16 = 3,
    Int64 = 4,
    Int32 = 5,
    Int16 = 6,
    Int8 = 7,
    UInt64 = 8,
    UInt32 = 9,
    UInt16 = 10,
    UInt8 = 11,
    Bool = 12,
}

impl DTypeId {
    pub fn element_size(&self) -> usize {
        match self {
            DTypeId::Float64 | DTypeId::Int64 | DTypeId::UInt64 => 8,
            DTypeId::Float32 | DTypeId::Int32 | DTypeId::UInt32 => 4,
            DTypeId::Float16 | DTypeId::BFloat16 | DTypeId::Int16 | DTypeId::UInt16 => 2,
            DTypeId::Int8 | DTypeId::UInt8 | DTypeId::Bool => 1,
        }
    }

    pub fn from_u8(val: u8) -> Result<Self, TpxError> {
        match val {
            0 => Ok(DTypeId::Float64),
            1 => Ok(DTypeId::Float32),
            2 => Ok(DTypeId::Float16),
            3 => Ok(DTypeId::BFloat16),
            4 => Ok(DTypeId::Int64),
            5 => Ok(DTypeId::Int32),
            6 => Ok(DTypeId::Int16),
            7 => Ok(DTypeId::Int8),
            8 => Ok(DTypeId::UInt64),
            9 => Ok(DTypeId::UInt32),
            10 => Ok(DTypeId::UInt16),
            11 => Ok(DTypeId::UInt8),
            12 => Ok(DTypeId::Bool),
            other => Err(TpxError::InvalidDType(other)),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            DTypeId::Float64 => "float64",
            DTypeId::Float32 => "float32",
            DTypeId::Float16 => "float16",
            DTypeId::BFloat16 => "bfloat16",
            DTypeId::Int64 => "int64",
            DTypeId::Int32 => "int32",
            DTypeId::Int16 => "int16",
            DTypeId::Int8 => "int8",
            DTypeId::UInt64 => "uint64",
            DTypeId::UInt32 => "uint32",
            DTypeId::UInt16 => "uint16",
            DTypeId::UInt8 => "uint8",
            DTypeId::Bool => "bool",
        }
    }
}

pub const ALL_DTYPES: &[DTypeId] = &[
    DTypeId::Float64,
    DTypeId::Float32,
    DTypeId::Float16,
    DTypeId::BFloat16,
    DTypeId::Int64,
    DTypeId::Int32,
    DTypeId::Int16,
    DTypeId::Int8,
    DTypeId::UInt64,
    DTypeId::UInt32,
    DTypeId::UInt16,
    DTypeId::UInt8,
    DTypeId::Bool,
];

pub const INT_DTYPES: &[DTypeId] = &[
    DTypeId::Int64,
    DTypeId::Int32,
    DTypeId::Int16,
    DTypeId::Int8,
    DTypeId::UInt64,
    DTypeId::UInt32,
    DTypeId::UInt16,
    DTypeId::UInt8,
];

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FilterId {
    None = 0,
    ByteShuffle = 1,
    BitShuffle = 2,
    DeltaZigZag = 3,
}

impl FilterId {
    pub fn from_u8(val: u8) -> Result<Self, TpxError> {
        match val {
            0 => Ok(FilterId::None),
            1 => Ok(FilterId::ByteShuffle),
            2 => Ok(FilterId::BitShuffle),
            3 => Ok(FilterId::DeltaZigZag),
            other => Err(TpxError::InvalidFilter(other)),
        }
    }
}

pub const ALL_FILTERS: &[FilterId] = &[
    FilterId::None,
    FilterId::ByteShuffle,
    FilterId::BitShuffle,
    FilterId::DeltaZigZag,
];

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CodecId {
    Raw = 0,
    Chimp128 = 1,
    Zstd = 2,
    Lz4 = 3,
}

impl CodecId {
    pub fn from_u8(val: u8) -> Result<Self, TpxError> {
        match val {
            0 => Ok(CodecId::Raw),
            1 => Ok(CodecId::Chimp128),
            2 => Ok(CodecId::Zstd),
            3 => Ok(CodecId::Lz4),
            other => Err(TpxError::InvalidCodec(other)),
        }
    }
}

pub const ALL_CODECS: &[CodecId] = &[CodecId::Raw, CodecId::Chimp128, CodecId::Zstd, CodecId::Lz4];

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttrTypeId {
    String = 0,
    Int64 = 1,
    Float64 = 2,
    Bool = 3,
    Bytes = 4,
}

impl AttrTypeId {
    pub fn from_u8(val: u8) -> Result<Self, TpxError> {
        match val {
            0 => Ok(AttrTypeId::String),
            1 => Ok(AttrTypeId::Int64),
            2 => Ok(AttrTypeId::Float64),
            3 => Ok(AttrTypeId::Bool),
            4 => Ok(AttrTypeId::Bytes),
            other => Err(TpxError::InvalidAttrType(other)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenMode {
    ReadOnly,
    ReadWrite,
    Append,
}

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkRef {
    pub shard_id: u16,
    pub chunk_offset: u64,
    pub chunk_len: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OwnedTensor {
    pub dtype: DTypeId,
    pub shape: Vec<u32>,
    pub data: Vec<u8>,
}

impl OwnedTensor {
    pub fn new(dtype: DTypeId, shape: Vec<u32>, data: Vec<u8>) -> Self {
        Self { dtype, shape, data }
    }

    pub fn total_elements(&self) -> usize {
        if self.shape.is_empty() {
            return 0;
        }
        self.shape.iter().map(|&dim| dim as usize).product()
    }

    pub fn expected_byte_len(&self) -> usize {
        self.total_elements() * self.dtype.element_size()
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum TpxError {
    #[error("invalid magic: expected {expected:?}, got {got:?}")]
    InvalidMagic { expected: [u8; 4], got: [u8; 4] },

    #[error("unsupported format version: {0:#06x}")]
    UnsupportedVersion(u16),

    #[error("invalid checksum: expected {expected:#x}, got {got:#x}")]
    ChecksumMismatch { expected: u64, got: u64 },

    #[error("chunk CRC32C mismatch at offset {offset}: expected {expected:#x}, got {got:#x}")]
    ChunkCrcMismatch {
        offset: u64,
        expected: u32,
        got: u32,
    },

    #[error("invalid end magic in footer: {0:?}")]
    InvalidEndMagic([u8; 4]),

    #[error("key not found: {0}")]
    KeyNotFound(String),

    #[error("version {version} not found for key {key}")]
    VersionNotFound { key: String, version: u32 },

    #[error("invalid dtype identifier: {0}")]
    InvalidDType(u8),

    #[error("invalid filter identifier: {0}")]
    InvalidFilter(u8),

    #[error("invalid codec identifier: {0}")]
    InvalidCodec(u8),

    #[error("invalid attribute type identifier: {0}")]
    InvalidAttrType(u8),

    #[error("attribute value too large: {len} bytes exceeds max 4096 bytes")]
    AttrValueTooLarge { len: usize },

    #[error("buffer too small for {component}: required {required} bytes, got {provided} bytes")]
    BufferTooSmall {
        component: &'static str,
        required: usize,
        provided: usize,
    },

    #[error("decompression length mismatch: expected {expected} bytes, got {got} bytes")]
    DecompressLengthMismatch { expected: usize, got: usize },

    #[error("codec error: {0}")]
    CodecError(String),

    #[error("shard {0} out of range")]
    ShardOutOfRange(u16),

    #[error("file is locked by another writer")]
    AlreadyLocked,

    #[error("file opened in read-only mode")]
    ReadOnlyMode,

    #[error("feature not enabled or available on this platform: {0}")]
    UnsupportedFeature(&'static str),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("corrupted archive: {0}")]
    CorruptArchive(String),
}

impl fmt::Display for OwnedTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OwnedTensor(dtype={}, shape={:?}, bytes={})",
            self.dtype.as_str(),
            self.shape,
            self.data.len()
        )
    }
}
