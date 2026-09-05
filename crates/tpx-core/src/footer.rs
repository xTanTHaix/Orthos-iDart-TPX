use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};

use crate::types::{TpxError, FOOTER_SIZE, MAGIC_END_CHECKPOINT, MAGIC_END_FINAL};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Footer {
    pub toc_offset: u64,
    pub total_chunks: u32,
    pub checksum: u64,
    pub magic_end: [u8; 4],
}

impl Default for Footer {
    fn default() -> Self {
        Self {
            toc_offset: 0,
            total_chunks: 0,
            checksum: 0,
            magic_end: MAGIC_END_FINAL,
        }
    }
}

impl Footer {
    pub fn new(toc_offset: u64, total_chunks: u32, checksum: u64, is_checkpoint: bool) -> Self {
        Self {
            toc_offset,
            total_chunks,
            checksum,
            magic_end: if is_checkpoint {
                MAGIC_END_CHECKPOINT
            } else {
                MAGIC_END_FINAL
            },
        }
    }

    pub fn is_checkpoint(&self) -> bool {
        self.magic_end == MAGIC_END_CHECKPOINT
    }

    pub fn is_final(&self) -> bool {
        self.magic_end == MAGIC_END_FINAL
    }

    pub fn serialize(&self) -> [u8; FOOTER_SIZE] {
        let mut buf = [0u8; FOOTER_SIZE];
        let mut cursor = Cursor::new(&mut buf[..]);

        cursor
            .write_u64::<LittleEndian>(self.toc_offset)
            .expect("toc_offset write");
        cursor
            .write_u32::<LittleEndian>(self.total_chunks)
            .expect("total_chunks write");
        cursor
            .write_u64::<LittleEndian>(self.checksum)
            .expect("checksum write");
        cursor.write_all(&self.magic_end).expect("magic_end write");

        buf
    }

    pub fn deserialize(buf: &[u8]) -> Result<Self, TpxError> {
        if buf.len() < FOOTER_SIZE {
            return Err(TpxError::BufferTooSmall {
                component: "Footer",
                required: FOOTER_SIZE,
                provided: buf.len(),
            });
        }

        let slice = &buf[buf.len() - FOOTER_SIZE..];
        let mut cursor = Cursor::new(slice);

        let toc_offset = cursor.read_u64::<LittleEndian>()?;
        let total_chunks = cursor.read_u32::<LittleEndian>()?;
        let checksum = cursor.read_u64::<LittleEndian>()?;
        let mut magic_end = [0u8; 4];
        cursor
            .read_exact(&mut magic_end)
            .map_err(|e| TpxError::Io(e))?;

        if magic_end != MAGIC_END_FINAL && magic_end != MAGIC_END_CHECKPOINT {
            return Err(TpxError::InvalidEndMagic(magic_end));
        }

        Ok(Self {
            toc_offset,
            total_chunks,
            checksum,
            magic_end,
        })
    }
}

pub trait ReadAt {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<usize>;
}

impl ReadAt for [u8] {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<usize> {
        let offset = offset as usize;
        if offset >= self.len() {
            return Ok(0);
        }
        let available = self.len() - offset;
        let to_read = available.min(buf.len());
        buf[..to_read].copy_from_slice(&self[offset..offset + to_read]);
        Ok(to_read)
    }
}

impl ReadAt for Vec<u8> {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<usize> {
        self.as_slice().read_at(offset, buf)
    }
}

/// Scan backward from EOF in chunk_align-sized steps to locate the most recent valid footer.
/// Returns (Footer, footer_offset) if found.
pub fn scan_for_footer<R: ReadAt + ?Sized>(
    reader: &R,
    file_len: u64,
    chunk_align: u32,
) -> Result<Option<(Footer, u64)>, TpxError> {
    if file_len < FOOTER_SIZE as u64 {
        return Ok(None);
    }

    let mut buf = [0u8; FOOTER_SIZE];

    // First check exact EOF - FOOTER_SIZE
    let eof_offset = file_len - FOOTER_SIZE as u64;
    let read_bytes = reader.read_at(eof_offset, &mut buf)?;
    if read_bytes == FOOTER_SIZE {
        if let Ok(footer) = Footer::deserialize(&buf) {
            if footer.toc_offset < eof_offset {
                return Ok(Some((footer, eof_offset)));
            }
        }
    }

    // If file is unaligned or corrupted at the very end, scan backward in steps of chunk_align
    let align = chunk_align.max(512) as u64;
    let mut current_offset = (file_len / align) * align;

    while current_offset >= FOOTER_SIZE as u64 {
        let probe_offset = current_offset - FOOTER_SIZE as u64;
        let read = reader.read_at(probe_offset, &mut buf)?;
        if read == FOOTER_SIZE {
            if let Ok(footer) = Footer::deserialize(&buf) {
                if footer.toc_offset < probe_offset {
                    return Ok(Some((footer, probe_offset)));
                }
            }
        }
        if current_offset <= align {
            break;
        }
        current_offset -= align;
    }

    Ok(None)
}
