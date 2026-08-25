//! Canonical padded RFC 4648 base64 with decode->re-encode identity.

use crate::err::{refuse, Code, Result};

const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(A[(n >> 18) as usize & 63] as char);
        out.push(A[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 {
            A[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            A[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Strict decode: padded, no whitespace, canonical (re-encode must reproduce the input).
pub fn decode(s: &str) -> Result<Vec<u8>> {
    let b = s.as_bytes();
    if !b.len().is_multiple_of(4) {
        return refuse(
            Code::BAD_BASE64,
            format!("length {} not a multiple of 4", b.len()),
        );
    }
    let val = |c: u8| -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a') as u32 + 26),
            b'0'..=b'9' => Some((c - b'0') as u32 + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    };
    let mut out = Vec::with_capacity(b.len() / 4 * 3);
    for (i, q) in b.chunks(4).enumerate() {
        let last = i == b.len() / 4 - 1;
        let pad = q.iter().filter(|c| **c == b'=').count();
        if pad > 2 || (pad > 0 && !last) {
            return refuse(Code::BAD_BASE64, "misplaced padding");
        }
        let mut n = 0u32;
        for (j, c) in q.iter().enumerate() {
            let v = if *c == b'=' {
                if j < 4 - pad {
                    return refuse(Code::BAD_BASE64, "padding inside a quantum");
                }
                0
            } else {
                match val(*c) {
                    Some(v) => v,
                    None => {
                        return refuse(Code::BAD_BASE64, format!("illegal byte {:?}", *c as char))
                    }
                }
            };
            n = (n << 6) | v;
        }
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
    }
    if encode(&out) != s {
        return refuse(Code::BAD_BASE64, "non-canonical base64 (re-encode differs)");
    }
    Ok(out)
}
