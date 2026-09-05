use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Cursor;

use crate::types::{CodecId, FilterId, TpxError, BLOCK_ENTRY_SIZE, CHUNK_HEADER_SIZE};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockEntry {
    pub compressed_len: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkHeader {
    pub codec_id: CodecId,
    pub filter_id: FilterId,
    pub block_count: u16,
    pub block_size: u32,
    pub uncompressed_len: u32,
    pub payload_crc32c: u32,
}

impl ChunkHeader {
    pub fn new(
        codec_id: CodecId,
        filter_id: FilterId,
        block_count: u16,
        block_size: u32,
        uncompressed_len: u32,
        payload_crc32c: u32,
    ) -> Self {
        Self {
            codec_id,
            filter_id,
            block_count,
            block_size,
            uncompressed_len,
            payload_crc32c,
        }
    }

    pub fn serialize(&self, block_table: &[BlockEntry]) -> Vec<u8> {
        let total_size = CHUNK_HEADER_SIZE + block_table.len() * BLOCK_ENTRY_SIZE;
        let mut buf = Vec::with_capacity(total_size);

        // 16-byte ChunkHeader (little-endian)
        buf.write_u8(self.codec_id as u8).expect("codec_id write");
        buf.write_u8(self.filter_id as u8).expect("filter_id write");
        buf.write_u16::<LittleEndian>(self.block_count)
            .expect("block_count write");
        buf.write_u32::<LittleEndian>(self.block_size)
            .expect("block_size write");
        buf.write_u32::<LittleEndian>(self.uncompressed_len)
            .expect("uncompressed_len write");
        buf.write_u32::<LittleEndian>(self.payload_crc32c)
            .expect("payload_crc32c write");

        for entry in block_table {
            buf.write_u32::<LittleEndian>(entry.compressed_len)
                .expect("compressed_len write");
        }

        buf
    }

    pub fn deserialize(buf: &[u8]) -> Result<(Self, Vec<BlockEntry>), TpxError> {
        if buf.len() < CHUNK_HEADER_SIZE {
            return Err(TpxError::BufferTooSmall {
                component: "ChunkHeader",
                required: CHUNK_HEADER_SIZE,
                provided: buf.len(),
            });
        }

        let mut cursor = Cursor::new(buf);
        let codec_id_u8 = cursor.read_u8().map_err(TpxError::Io)?;
        let codec_id = CodecId::from_u8(codec_id_u8)?;

        let filter_id_u8 = cursor.read_u8().map_err(TpxError::Io)?;
        let filter_id = FilterId::from_u8(filter_id_u8)?;

        let block_count = cursor.read_u16::<LittleEndian>()?;
        let block_size = cursor.read_u32::<LittleEndian>()?;
        let uncompressed_len = cursor.read_u32::<LittleEndian>()?;
        let payload_crc32c = cursor.read_u32::<LittleEndian>()?;

        let header = Self {
            codec_id,
            filter_id,
            block_count,
            block_size,
            uncompressed_len,
            payload_crc32c,
        };

        let required_total = CHUNK_HEADER_SIZE + (block_count as usize) * BLOCK_ENTRY_SIZE;
        if buf.len() < required_total {
            return Err(TpxError::BufferTooSmall {
                component: "BlockTable",
                required: required_total,
                provided: buf.len(),
            });
        }

        let mut block_table = Vec::with_capacity(block_count as usize);
        for _ in 0..block_count {
            let compressed_len = cursor.read_u32::<LittleEndian>()?;
            block_table.push(BlockEntry { compressed_len });
        }

        Ok((header, block_table))
    }
}

pub fn pad_to_align(buf: &mut Vec<u8>, chunk_align: u32) {
    if chunk_align == 0 {
        return;
    }
    let align = chunk_align as usize;
    let remainder = buf.len() % align;
    if remainder != 0 {
        let padding = align - remainder;
        buf.resize(buf.len() + padding, 0);
    }
}
