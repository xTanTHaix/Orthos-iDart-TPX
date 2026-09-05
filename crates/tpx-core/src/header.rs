use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Cursor;

use crate::types::{
    TpxError, DEFAULT_BLOCK_SIZE, DEFAULT_CHUNK_ALIGN, FLAG_CHECKSUMMED, HEADER_SIZE, MAGIC,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileHeader {
    pub magic: [u8; 4],
    pub format_version: u16,
    pub flags: u16,
    pub chunk_align: u32,
    pub default_block_size: u32,
    pub shard_count: u16,
    pub reserved: u16,
    pub global_attrs_offset: u64,
    pub global_attrs_len: u64,
    pub shard_table_offset: u64,
    pub reserved2: u32,
}

impl Default for FileHeader {
    fn default() -> Self {
        Self {
            magic: MAGIC,
            format_version: 0x0100, // v1.0
            flags: FLAG_CHECKSUMMED,
            chunk_align: DEFAULT_CHUNK_ALIGN,
            default_block_size: DEFAULT_BLOCK_SIZE,
            shard_count: 1,
            reserved: 0,
            global_attrs_offset: 0,
            global_attrs_len: 0,
            shard_table_offset: 0,
            reserved2: 0,
        }
    }
}

impl FileHeader {
    pub fn new(shard_count: u16, chunk_align: u32, flags: u16) -> Self {
        Self {
            magic: MAGIC,
            format_version: 0x0100,
            flags,
            chunk_align,
            default_block_size: DEFAULT_BLOCK_SIZE,
            shard_count: shard_count.max(1),
            reserved: 0,
            global_attrs_offset: 0,
            global_attrs_len: 0,
            shard_table_offset: 0,
            reserved2: 0,
        }
    }

    pub fn serialize(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        let mut cursor = Cursor::new(&mut buf[..]);

        cursor.write_all(&self.magic).expect("magic write");
        cursor
            .write_u16::<LittleEndian>(self.format_version)
            .expect("format_version write");
        cursor
            .write_u16::<LittleEndian>(self.flags)
            .expect("flags write");
        cursor
            .write_u32::<LittleEndian>(self.chunk_align)
            .expect("chunk_align write");
        cursor
            .write_u32::<LittleEndian>(self.default_block_size)
            .expect("default_block_size write");
        cursor
            .write_u16::<LittleEndian>(self.shard_count)
            .expect("shard_count write");
        cursor
            .write_u16::<LittleEndian>(self.reserved)
            .expect("reserved write");
        cursor
            .write_u64::<LittleEndian>(self.global_attrs_offset)
            .expect("global_attrs_offset write");
        cursor
            .write_u64::<LittleEndian>(self.global_attrs_len)
            .expect("global_attrs_len write");
        cursor
            .write_u64::<LittleEndian>(self.shard_table_offset)
            .expect("shard_table_offset write");
        cursor
            .write_u32::<LittleEndian>(self.reserved2)
            .expect("reserved2 write");

        buf
    }

    pub fn deserialize(buf: &[u8]) -> Result<Self, TpxError> {
        if buf.len() < HEADER_SIZE {
            return Err(TpxError::BufferTooSmall {
                component: "FileHeader",
                required: HEADER_SIZE,
                provided: buf.len(),
            });
        }

        let mut cursor = Cursor::new(buf);
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic).map_err(|e| TpxError::Io(e))?;

        let format_version = cursor.read_u16::<LittleEndian>()?;
        let flags = cursor.read_u16::<LittleEndian>()?;
        let chunk_align = cursor.read_u32::<LittleEndian>()?;
        let default_block_size = cursor.read_u32::<LittleEndian>()?;
        let shard_count = cursor.read_u16::<LittleEndian>()?;
        let reserved = cursor.read_u16::<LittleEndian>()?;
        let global_attrs_offset = cursor.read_u64::<LittleEndian>()?;
        let global_attrs_len = cursor.read_u64::<LittleEndian>()?;
        let shard_table_offset = cursor.read_u64::<LittleEndian>()?;
        let reserved2 = cursor.read_u32::<LittleEndian>()?;

        let header = Self {
            magic,
            format_version,
            flags,
            chunk_align,
            default_block_size,
            shard_count,
            reserved,
            global_attrs_offset,
            global_attrs_len,
            shard_table_offset,
            reserved2,
        };

        header.validate()?;
        Ok(header)
    }

    pub fn validate(&self) -> Result<(), TpxError> {
        if self.magic != MAGIC {
            return Err(TpxError::InvalidMagic {
                expected: MAGIC,
                got: self.magic,
            });
        }

        let major_version = (self.format_version >> 8) as u8;
        if major_version > 1 {
            return Err(TpxError::UnsupportedVersion(self.format_version));
        }

        Ok(())
    }
}

use std::io::{Read, Write};
