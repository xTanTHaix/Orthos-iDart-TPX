use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Cursor;
use tpx_core::types::TpxError;

/// Chimp128 float compression algorithm (VLDB 2022).
/// Efficiently compresses IEEE-754 floats by XOR-delta with leading/trailing zero suppression
/// and a 128-entry history ring buffer.

pub struct Chimp128Encoder {
    history: [u64; 128],
    history_idx: usize,
    prev_val: u64,
    prev_leading_zeros: u8,
    prev_trailing_zeros: u8,
    bits_buffer: u64,
    bits_count: u8,
    output: Vec<u8>,
    count: u32,
}

impl Default for Chimp128Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Chimp128Encoder {
    pub fn new() -> Self {
        Self {
            history: [0u64; 128],
            history_idx: 0,
            prev_val: 0,
            prev_leading_zeros: u8::MAX,
            prev_trailing_zeros: u8::MAX,
            bits_buffer: 0,
            bits_count: 0,
            output: Vec::new(),
            count: 0,
        }
    }

    fn write_bits(&mut self, value: u64, num_bits: u8) {
        if num_bits == 0 {
            return;
        }

        for i in (0..num_bits).rev() {
            let bit = ((value >> i) & 1) as u8;
            self.bits_buffer = (self.bits_buffer << 1) | (bit as u64);
            self.bits_count += 1;

            if self.bits_count == 8 {
                self.output.push(self.bits_buffer as u8);
                self.bits_buffer = 0;
                self.bits_count = 0;
            }
        }
    }

    pub fn encode_f32(&mut self, val: f32) {
        let bits = (val.to_bits() as u64) << 32;
        self.encode_u64_internal(bits);
        self.count += 1;
    }

    pub fn encode_f64(&mut self, val: f64) {
        let bits = val.to_bits();
        self.encode_u64_internal(bits);
        self.count += 1;
    }

    fn encode_u64_internal(&mut self, val: u64) {
        if self.count == 0 {
            self.prev_val = val;
            self.history[0] = val;
            self.history_idx = 1;
            self.write_bits(val, 64);
            return;
        }

        let xor = val ^ self.prev_val;
        if xor == 0 {
            // Case 00: exact match with previous
            self.write_bits(0b00, 2);
        } else {
            // Check history for a better match
            let mut best_xor = xor;
            let mut best_idx: Option<usize> = None;

            for i in 0..128 {
                let candidate_xor = val ^ self.history[i];
                if candidate_xor.leading_zeros() > best_xor.leading_zeros() {
                    best_xor = candidate_xor;
                    best_idx = Some(i);
                }
            }

            if let Some(idx) = best_idx {
                // Case 01: match from history buffer
                self.write_bits(0b01, 2);
                self.write_bits(idx as u64, 7);
                self.encode_xor_payload(best_xor);
            } else {
                // Case 1: delta with previous
                self.write_bits(0b1, 1);
                self.encode_xor_payload(xor);
            }
        }

        self.prev_val = val;
        self.history[self.history_idx % 128] = val;
        self.history_idx += 1;
    }

    fn encode_xor_payload(&mut self, xor: u64) {
        let (leading_zeros, trailing_zeros, significant_bits, val_to_write) = if xor == 0 {
            (63u8, 0u8, 1u8, 0u64)
        } else {
            let lz = (xor.leading_zeros().min(63)) as u8;
            let tz = (xor.trailing_zeros().min(63)) as u8;
            let sig = (64u8.saturating_sub(lz).saturating_sub(tz)).max(1);
            let val = xor >> tz;
            (lz, tz, sig, val)
        };

        if self.prev_leading_zeros != u8::MAX
            && leading_zeros == self.prev_leading_zeros
            && trailing_zeros == self.prev_trailing_zeros
        {
            // Case 0: reuse previous zero bounds
            self.write_bits(0b0, 1);
            self.write_bits(val_to_write, significant_bits);
        } else {
            // Case 1: store new zero bounds
            self.write_bits(0b1, 1);
            self.write_bits(leading_zeros as u64, 6);
            self.write_bits(trailing_zeros as u64, 6);
            self.write_bits(val_to_write, significant_bits);
            self.prev_leading_zeros = leading_zeros;
            self.prev_trailing_zeros = trailing_zeros;
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        if self.bits_count > 0 {
            let shift = 8 - self.bits_count;
            let byte = (self.bits_buffer << shift) as u8;
            self.output.push(byte);
            self.bits_buffer = 0;
            self.bits_count = 0;
        }

        let mut final_buf = Vec::with_capacity(4 + self.output.len());
        final_buf.write_u32::<LittleEndian>(self.count).unwrap();
        final_buf.extend_from_slice(&self.output);
        final_buf
    }
}

pub struct Chimp128Decoder<'a> {
    cursor: Cursor<&'a [u8]>,
    history: [u64; 128],
    history_idx: usize,
    prev_val: u64,
    prev_leading_zeros: u8,
    prev_trailing_zeros: u8,
    current_byte: u8,
    bits_in_byte_left: u8,
    total_count: u32,
    decoded_count: u32,
}

impl<'a> Chimp128Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self, TpxError> {
        if data.len() < 4 {
            return Err(TpxError::BufferTooSmall {
                component: "Chimp128Decoder",
                required: 4,
                provided: data.len(),
            });
        }

        let mut cursor = Cursor::new(data);
        let total_count = cursor.read_u32::<LittleEndian>()?;

        Ok(Self {
            cursor,
            history: [0u64; 128],
            history_idx: 0,
            prev_val: 0,
            prev_leading_zeros: u8::MAX,
            prev_trailing_zeros: u8::MAX,
            current_byte: 0,
            bits_in_byte_left: 0,
            total_count,
            decoded_count: 0,
        })
    }

    fn read_bit(&mut self) -> Result<u64, TpxError> {
        if self.bits_in_byte_left == 0 {
            let mut byte_buf = [0u8; 1];
            use std::io::Read;
            if self.cursor.read_exact(&mut byte_buf).is_err() {
                self.current_byte = 0;
            } else {
                self.current_byte = byte_buf[0];
            }
            self.bits_in_byte_left = 8;
        }

        self.bits_in_byte_left -= 1;
        let bit = ((self.current_byte >> self.bits_in_byte_left) & 1) as u64;
        Ok(bit)
    }

    fn read_bits(&mut self, num_bits: u8) -> Result<u64, TpxError> {
        let mut result = 0u64;
        for _ in 0..num_bits {
            let bit = self.read_bit()?;
            result = (result << 1) | bit;
        }
        Ok(result)
    }

    pub fn decode_f32(&mut self) -> Option<f32> {
        if self.decoded_count >= self.total_count {
            return None;
        }
        let u64_val = self.decode_u64_internal().ok()?;
        self.decoded_count += 1;
        let bits = (u64_val >> 32) as u32;
        Some(f32::from_bits(bits))
    }

    pub fn decode_f64(&mut self) -> Option<f64> {
        if self.decoded_count >= self.total_count {
            return None;
        }
        let u64_val = self.decode_u64_internal().ok()?;
        self.decoded_count += 1;
        Some(f64::from_bits(u64_val))
    }

    fn decode_u64_internal(&mut self) -> Result<u64, TpxError> {
        if self.decoded_count == 0 {
            let val = self.read_bits(64)?;
            self.prev_val = val;
            self.history[0] = val;
            self.history_idx = 1;
            return Ok(val);
        }

        let first_bit = self.read_bits(1)?;
        let val = if first_bit == 0 {
            let second_bit = self.read_bits(1)?;
            if second_bit == 0 {
                self.prev_val
            } else {
                let hist_idx = self.read_bits(7)? as usize;
                let base_val = self.history[hist_idx % 128];
                let xor = self.decode_xor_payload()?;
                base_val ^ xor
            }
        } else {
            let xor = self.decode_xor_payload()?;
            self.prev_val ^ xor
        };

        self.prev_val = val;
        self.history[self.history_idx % 128] = val;
        self.history_idx += 1;
        Ok(val)
    }

    fn decode_xor_payload(&mut self) -> Result<u64, TpxError> {
        let is_new_zeros = self.read_bits(1)? == 1;
        let (leading_zeros, trailing_zeros) = if is_new_zeros {
            let lz = self.read_bits(6)? as u8;
            let tz = self.read_bits(6)? as u8;
            self.prev_leading_zeros = lz;
            self.prev_trailing_zeros = tz;
            (lz, tz)
        } else {
            (self.prev_leading_zeros, self.prev_trailing_zeros)
        };

        let significant_bits = (64u8
            .saturating_sub(leading_zeros)
            .saturating_sub(trailing_zeros))
        .max(1);
        let meaningful_bits = self.read_bits(significant_bits)?;
        let xor = meaningful_bits << trailing_zeros;
        Ok(xor)
    }
}
