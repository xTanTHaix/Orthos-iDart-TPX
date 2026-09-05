use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;
use std::io::{Cursor, Read, Write};

use crate::types::{AttrTypeId, TpxError};

pub const MAX_ATTR_VALUE_SIZE: usize = 4096;

#[derive(Clone, Debug, PartialEq)]
pub enum AttrValue {
    String(String),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    Bytes(Vec<u8>),
}

impl AttrValue {
    pub fn type_id(&self) -> AttrTypeId {
        match self {
            AttrValue::String(_) => AttrTypeId::String,
            AttrValue::Int64(_) => AttrTypeId::Int64,
            AttrValue::Float64(_) => AttrTypeId::Float64,
            AttrValue::Bool(_) => AttrTypeId::Bool,
            AttrValue::Bytes(_) => AttrTypeId::Bytes,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Attrs {
    pub entries: HashMap<String, AttrValue>,
}

pub type GlobalAttrs = Attrs;

impl Attrs {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn set(&mut self, key: impl Into<String>, value: AttrValue) -> Result<(), TpxError> {
        let val_len = match &value {
            AttrValue::String(s) => s.len(),
            AttrValue::Int64(_) => 8,
            AttrValue::Float64(_) => 8,
            AttrValue::Bool(_) => 1,
            AttrValue::Bytes(b) => b.len(),
        };

        if val_len > MAX_ATTR_VALUE_SIZE {
            return Err(TpxError::AttrValueTooLarge { len: val_len });
        }

        self.entries.insert(key.into(), value);
        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<&AttrValue> {
        self.entries.get(key)
    }

    pub fn remove(&mut self, key: &str) -> Option<AttrValue> {
        self.entries.remove(key)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Sort keys for deterministic serialized output
        let mut keys: Vec<&String> = self.entries.keys().collect();
        keys.sort();

        for key in keys {
            let val = &self.entries[key];
            let key_bytes = key.as_bytes();
            let key_len = (key_bytes.len().min(255)) as u8;

            buf.write_u8(key_len).expect("key_len write");
            buf.write_all(&key_bytes[..key_len as usize])
                .expect("key_bytes write");
            buf.write_u8(val.type_id() as u8).expect("type_id write");

            match val {
                AttrValue::String(s) => {
                    let s_bytes = s.as_bytes();
                    buf.write_u16::<LittleEndian>(s_bytes.len() as u16)
                        .expect("val_len write");
                    buf.write_all(s_bytes).expect("val write");
                }
                AttrValue::Int64(i) => {
                    buf.write_u16::<LittleEndian>(8).expect("val_len write");
                    buf.write_i64::<LittleEndian>(*i).expect("val write");
                }
                AttrValue::Float64(f) => {
                    buf.write_u16::<LittleEndian>(8).expect("val_len write");
                    buf.write_f64::<LittleEndian>(*f).expect("val write");
                }
                AttrValue::Bool(b) => {
                    buf.write_u16::<LittleEndian>(1).expect("val_len write");
                    buf.write_u8(if *b { 1 } else { 0 }).expect("val write");
                }
                AttrValue::Bytes(bytes) => {
                    buf.write_u16::<LittleEndian>(bytes.len() as u16)
                        .expect("val_len write");
                    buf.write_all(bytes).expect("val write");
                }
            }
        }

        buf
    }

    pub fn deserialize(buf: &[u8]) -> Result<Self, TpxError> {
        let mut cursor = Cursor::new(buf);
        let mut attrs = Self::new();
        let total_len = buf.len() as u64;

        while cursor.position() < total_len {
            let key_len = match cursor.read_u8() {
                Ok(len) => len as usize,
                Err(_) => break,
            };

            let mut key_bytes = vec![0u8; key_len];
            cursor
                .read_exact(&mut key_bytes)
                .map_err(|e| TpxError::Io(e))?;
            let key = String::from_utf8(key_bytes)
                .map_err(|e| TpxError::CorruptArchive(e.to_string()))?;

            let type_id_u8 = cursor.read_u8().map_err(|e| TpxError::Io(e))?;
            let type_id = AttrTypeId::from_u8(type_id_u8)?;

            let val_len = cursor
                .read_u16::<LittleEndian>()
                .map_err(|e| TpxError::Io(e))? as usize;

            if val_len > MAX_ATTR_VALUE_SIZE {
                return Err(TpxError::AttrValueTooLarge { len: val_len });
            }

            let mut val_bytes = vec![0u8; val_len];
            cursor
                .read_exact(&mut val_bytes)
                .map_err(|e| TpxError::Io(e))?;

            let val = match type_id {
                AttrTypeId::String => {
                    let s = String::from_utf8(val_bytes)
                        .map_err(|e| TpxError::CorruptArchive(e.to_string()))?;
                    AttrValue::String(s)
                }
                AttrTypeId::Int64 => {
                    if val_len != 8 {
                        return Err(TpxError::CorruptArchive("Int64 attr len != 8".into()));
                    }
                    let num = Cursor::new(&val_bytes).read_i64::<LittleEndian>()?;
                    AttrValue::Int64(num)
                }
                AttrTypeId::Float64 => {
                    if val_len != 8 {
                        return Err(TpxError::CorruptArchive("Float64 attr len != 8".into()));
                    }
                    let num = Cursor::new(&val_bytes).read_f64::<LittleEndian>()?;
                    AttrValue::Float64(num)
                }
                AttrTypeId::Bool => {
                    if val_len != 1 {
                        return Err(TpxError::CorruptArchive("Bool attr len != 1".into()));
                    }
                    AttrValue::Bool(val_bytes[0] != 0)
                }
                AttrTypeId::Bytes => AttrValue::Bytes(val_bytes),
            };

            attrs.set(key, val)?;
        }

        Ok(attrs)
    }
}
