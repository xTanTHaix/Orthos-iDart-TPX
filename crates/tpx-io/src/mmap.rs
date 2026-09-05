use fs2::FileExt;
use parking_lot::{Mutex, RwLock};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tpx_core::types::{OpenMode, TpxError};

use crate::tier::{IoBackend, IoHints, IoTier};

const WRITE_BUFFER_LIMIT: usize = 1024 * 1024; // 1 MB write buffer coalescing for direct storage

pub struct MmapBackend {
    file: RwLock<File>,
    write_buffer: Mutex<Vec<u8>>,
    write_buffer_start: AtomicU64,
    logical_len: AtomicU64,
    pub path: PathBuf,
    mode: OpenMode,
}

impl MmapBackend {
    pub fn open_or_create(
        path: &Path,
        mode: OpenMode,
        _hints: &IoHints,
    ) -> Result<Box<dyn IoBackend>, TpxError> {
        let file = match mode {
            OpenMode::ReadOnly => OpenOptions::new()
                .read(true)
                .open(path)
                .map_err(TpxError::Io)?,
            OpenMode::ReadWrite | OpenMode::Append => {
                let f = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .open(path)
                    .map_err(TpxError::Io)?;
                // Apply advisory exclusive file lock for SWMR writer
                if f.try_lock_exclusive().is_err() {
                    return Err(TpxError::AlreadyLocked);
                }
                f
            }
        };

        let initial_len = file.metadata().map(|m| m.len()).unwrap_or(0);

        Ok(Box::new(Self {
            file: RwLock::new(file),
            write_buffer: Mutex::new(Vec::with_capacity(WRITE_BUFFER_LIMIT)),
            write_buffer_start: AtomicU64::new(initial_len),
            logical_len: AtomicU64::new(initial_len),
            path: path.to_path_buf(),
            mode,
        }))
    }

    fn flush_buffer_internal(&self) -> Result<(), TpxError> {
        let mut wb = self.write_buffer.lock();
        if wb.is_empty() {
            return Ok(());
        }
        let mut file = self.file.write();
        let target_start = self.write_buffer_start.load(Ordering::SeqCst);
        file.seek(SeekFrom::Start(target_start))
            .map_err(TpxError::Io)?;
        file.write_all(&wb).map_err(TpxError::Io)?;
        let new_start = target_start + wb.len() as u64;
        self.write_buffer_start.store(new_start, Ordering::SeqCst);
        wb.clear();
        Ok(())
    }
}

impl IoBackend for MmapBackend {
    fn write_at(&self, offset: u64, data: &[u8]) -> Result<(), TpxError> {
        if self.mode == OpenMode::ReadOnly {
            return Err(TpxError::ReadOnlyMode);
        }

        self.flush_buffer_internal()?;

        let mut file = self.file.write();
        file.seek(SeekFrom::Start(offset)).map_err(TpxError::Io)?;
        file.write_all(data).map_err(TpxError::Io)?;
        let end = offset + data.len() as u64;
        self.logical_len.fetch_max(end, Ordering::SeqCst);
        self.write_buffer_start.fetch_max(end, Ordering::SeqCst);
        Ok(())
    }

    fn read_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, TpxError> {
        let mut buf = vec![0u8; len];
        self.read_at_into(offset, &mut buf)?;
        Ok(buf)
    }

    fn read_at_into(&self, offset: u64, buf: &mut [u8]) -> Result<(), TpxError> {
        let wb_start = self.write_buffer_start.load(Ordering::SeqCst);
        if offset + buf.len() as u64 > wb_start {
            self.flush_buffer_internal()?;
        }

        let mut file = self.file.write();
        file.seek(SeekFrom::Start(offset)).map_err(TpxError::Io)?;
        file.read_exact(buf).map_err(TpxError::Io)?;
        Ok(())
    }

    fn append(&self, data: &[u8]) -> Result<u64, TpxError> {
        if self.mode == OpenMode::ReadOnly {
            return Err(TpxError::ReadOnlyMode);
        }

        let mut wb = self.write_buffer.lock();
        let append_offset = self
            .logical_len
            .fetch_add(data.len() as u64, Ordering::SeqCst);

        if data.len() >= WRITE_BUFFER_LIMIT {
            if !wb.is_empty() {
                let mut file = self.file.write();
                let target_start = self.write_buffer_start.load(Ordering::SeqCst);
                file.seek(SeekFrom::Start(target_start))
                    .map_err(TpxError::Io)?;
                file.write_all(&wb).map_err(TpxError::Io)?;
                wb.clear();
            }
            let mut file = self.file.write();
            file.seek(SeekFrom::Start(append_offset))
                .map_err(TpxError::Io)?;
            file.write_all(data).map_err(TpxError::Io)?;
            self.write_buffer_start
                .store(append_offset + data.len() as u64, Ordering::SeqCst);
            return Ok(append_offset);
        }

        wb.extend_from_slice(data);
        if wb.len() >= WRITE_BUFFER_LIMIT {
            let mut file = self.file.write();
            let target_start = self.write_buffer_start.load(Ordering::SeqCst);
            file.seek(SeekFrom::Start(target_start))
                .map_err(TpxError::Io)?;
            file.write_all(&wb).map_err(TpxError::Io)?;
            let new_start = target_start + wb.len() as u64;
            self.write_buffer_start.store(new_start, Ordering::SeqCst);
            wb.clear();
        }

        Ok(append_offset)
    }

    fn len(&self) -> u64 {
        self.logical_len.load(Ordering::SeqCst)
    }

    fn sync(&self) -> Result<(), TpxError> {
        if self.mode != OpenMode::ReadOnly {
            self.flush_buffer_internal()?;
            let file = self.file.read();
            file.sync_data().map_err(TpxError::Io)?;
        }
        Ok(())
    }

    fn truncate(&self, len: u64) -> Result<(), TpxError> {
        let mut wb = self.write_buffer.lock();
        wb.clear();
        self.write_buffer_start.store(len, Ordering::SeqCst);
        self.logical_len.store(len, Ordering::SeqCst);

        let file = self.file.write();
        file.set_len(len).map_err(TpxError::Io)?;
        Ok(())
    }

    fn tier(&self) -> IoTier {
        IoTier::Mmap
    }

    fn close(&self) -> Result<(), TpxError> {
        if self.mode != OpenMode::ReadOnly {
            let _ = self.flush_buffer_internal();
            let file = self.file.write();
            let _ = file.unlock();
            file.sync_all().map_err(TpxError::Io)?;
        }
        Ok(())
    }
}

impl Drop for MmapBackend {
    fn drop(&mut self) {
        if self.mode != OpenMode::ReadOnly {
            let _ = self.flush_buffer_internal();
            let file = self.file.write();
            let _ = file.unlock();
        }
    }
}
