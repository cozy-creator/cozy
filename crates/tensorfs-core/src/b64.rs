//! Canonical padded RFC 4648 base64: the `base64` crate's STANDARD engine, which requires
//! canonical padding and refuses trailing bits, so a decode re-encodes to its input.

use base64::{engine::general_purpose::STANDARD, Engine};

use crate::err::{Code, Refusal, Result};

pub fn encode(data: &[u8]) -> String {
    STANDARD.encode(data)
}

pub fn decode(s: &str) -> Result<Vec<u8>> {
    STANDARD.decode(s).map_err(|e| Refusal {
        code: Code::BAD_BASE64,
        detail: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_refuses_what_is_not_canonical() {
        for data in [&b""[..], b"f", b"fo", b"foo", b"foob", b"fooba", b"foobar"] {
            assert_eq!(decode(&encode(data)).unwrap(), data);
        }
        assert_eq!(encode(b"foobar"), "Zm9vYmFy");
        for bad in ["Zg", "Zg=", "Zh==", "Zm9v\n", " Zm9v", "Zm9v====", "Zm=9", "Z===", "Zm9vYg", "Zm9vYh==", "@@@@"] {
            assert_eq!(decode(bad).unwrap_err().code, Code::BAD_BASE64, "{bad:?}");
        }
    }
}
