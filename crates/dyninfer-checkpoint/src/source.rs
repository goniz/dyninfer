//! Random-access source abstraction for checkpoint inspection.
//!
//! Uses buffered file seeks rather than mmap so this crate can remain
//! `forbid(unsafe_code)`. Runtime parameter providers may mmap later in an
//! FFI-adjacent crate when wiring IREE external parameters.

use dyninfer_error::{DynInferError, Result};
use parking_lot::Mutex;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Byte-addressable source used by container readers.
pub trait RandomAccessSource: Send + Sync {
    fn path(&self) -> Option<&Path>;
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()>;

    /// Read a contiguous range into an owned buffer.
    fn read_range(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        let len_usize = usize::try_from(len)
            .map_err(|_| DynInferError::io("range length does not fit usize"))?;
        let mut buf = vec![0u8; len_usize];
        self.read_exact_at(offset, &mut buf)?;
        Ok(buf)
    }
}

/// Seekable file source with interior mutability for shared readers.
pub struct FileSource {
    path: PathBuf,
    len: u64,
    file: Mutex<File>,
}

impl FileSource {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|e| {
            DynInferError::io_path(path.display().to_string(), format!("open failed: {e}"))
        })?;
        let len = file
            .metadata()
            .map_err(|e| {
                DynInferError::io_path(path.display().to_string(), format!("metadata failed: {e}"))
            })?
            .len();
        Ok(Self {
            path,
            len,
            file: Mutex::new(file),
        })
    }

    pub fn into_arc(self) -> Arc<dyn RandomAccessSource> {
        Arc::new(self)
    }
}

impl RandomAccessSource for FileSource {
    fn path(&self) -> Option<&Path> {
        Some(&self.path)
    }

    fn len(&self) -> u64 {
        self.len
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let end = offset
            .checked_add(buf.len() as u64)
            .ok_or_else(|| DynInferError::io("byte range overflow"))?;
        if end > self.len {
            return Err(DynInferError::io_path(
                self.path.display().to_string(),
                format!(
                    "read past end: offset={offset} len={} file_len={}",
                    buf.len(),
                    self.len
                ),
            ));
        }
        let mut file = self.file.lock();
        file.seek(SeekFrom::Start(offset)).map_err(|e| {
            DynInferError::io_path(self.path.display().to_string(), format!("seek failed: {e}"))
        })?;
        file.read_exact(buf).map_err(|e| {
            DynInferError::io_path(self.path.display().to_string(), format!("read failed: {e}"))
        })?;
        Ok(())
    }
}

/// In-memory source for tests and small fixtures.
pub struct BytesSource {
    path: Option<PathBuf>,
    data: Arc<[u8]>,
}

impl BytesSource {
    pub fn new(data: impl Into<Arc<[u8]>>) -> Self {
        Self {
            path: None,
            data: data.into(),
        }
    }

    pub fn with_path(path: impl Into<PathBuf>, data: impl Into<Arc<[u8]>>) -> Self {
        Self {
            path: Some(path.into()),
            data: data.into(),
        }
    }

    pub fn into_arc(self) -> Arc<dyn RandomAccessSource> {
        Arc::new(self)
    }
}

impl RandomAccessSource for BytesSource {
    fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    fn len(&self) -> u64 {
        self.data.len() as u64
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let end = offset
            .checked_add(buf.len() as u64)
            .ok_or_else(|| DynInferError::io("byte range overflow"))?;
        if end > self.len() {
            return Err(DynInferError::io(format!(
                "read past end: offset={offset} len={} source_len={}",
                buf.len(),
                self.len()
            )));
        }
        let start = offset as usize;
        buf.copy_from_slice(&self.data[start..start + buf.len()]);
        Ok(())
    }
}
