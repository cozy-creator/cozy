//! The decode cores (spec §6, building blocks) against `conformance/vectors/cores.json`: affine and table cases
//! over the layouts producers use, and gguf-py's own q4_0 and q8_0 block structs.

use std::path::Path;

use base64::Engine;
use serde_json::Value;
use tensorfs::format::decode::affine::{dequant, Codes, Fill, Offset, Order, Packing};
use tensorfs::format::decode::grid::{Grid, ScaleLayout};
use tensorfs::format::decode::{blocks, table};

fn b64(v: &Value) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(v.as_str().unwrap())
        .unwrap()
}

fn f32s(v: &Value) -> Vec<f32> {
    b64(v)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn uint(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

fn packing<'a>(p: &Value, slots: &'a [u8]) -> Packing<'a> {
    let fill = match p["fill"].as_str().unwrap_or("word") {
        "word" => Fill::Word,
        "dense" => Fill::Dense,
        _ => Fill::Planar,
    };
    let order = match &p["order"] {
        Value::Array(_) => Order::Slots(slots),
        v => match v.as_str().unwrap_or("low_first") {
            "low_first" => Order::LowFirst,
            _ => Order::HighFirst,
        },
    };
    let codes = match p["signed"].as_bool().unwrap_or(false) {
        true => Codes::Signed,
        false => Codes::Unsigned(p["offset"].as_i64().unwrap_or(0) as i32),
    };
    Packing {
        bits: uint(&p["bits"]) as u32,
        word: uint(&p["word"]) as u32,
        transposed: p["transposed"].as_bool().unwrap_or(false),
        axis: p["axis"].as_u64().unwrap_or(1) as usize,
        fill,
        order,
        codes,
    }
}

fn grid<'a>(g: &Value, index: &'a [u32]) -> Grid<'a> {
    match g["kind"].as_str().unwrap() {
        "tensor" => Grid::Tensor,
        "row" => Grid::Row,
        "col" => Grid::Col,
        _ => Grid::Blocks {
            size: uint(&g["size"]),
            layout: ScaleLayout::parse(g["layout"].as_str().unwrap()).unwrap(),
            index: g.get("index").map(|_| index),
        },
    }
}

/// gguf's q4_0 `{f16 d; u8 qs[16]}` (element j in qs[j]'s low nibble, j + 16 in its high one,
/// `(q − 8) × d`) and q8_0 `{f16 d; i8 qs[32]}` (`q × d`).
fn gguf(kind: &str, block: &[u8], out: &mut [f32]) -> tensorfs::format::Result<()> {
    let (bits, fill, codes, zero) = match kind {
        "q4_0" => (4, Fill::Planar, Codes::Unsigned(0), Offset::Zero(&[8.0])),
        _ => (8, Fill::Word, Codes::Signed, Offset::None),
    };
    let p = Packing {
        bits,
        word: 8,
        transposed: false,
        axis: 1,
        fill,
        order: Order::LowFirst,
        codes,
    };
    let q = p.lane(&block[2..], out.len())?;
    out.copy_from_slice(&dequant(
        &q,
        &[blocks::f16_at(block, 0)],
        zero,
        &Grid::Tensor,
        1,
        out.len(),
    ));
    Ok(())
}

#[test]
fn cores_decode_every_case() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance/vectors/cores.json");
    let doc: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let mut seen = 0;
    for case in doc["cases"].as_array().unwrap() {
        let (rows, cols) = (uint(&case["rows"]), uint(&case["cols"]));
        let got = match case["core"].as_str().unwrap() {
            "blocks" => {
                let (n, size) = (uint(&case["elements"]), uint(&case["size"]));
                let kind = case["type"].as_str().unwrap();
                blocks::walk(rows, cols, &b64(&case["blocks"]), n, size, |b, out| {
                    gguf(kind, b, out)
                })
                .unwrap()
            }
            core => {
                let slots: Vec<u8> = (case["packing"]["order"].as_array().into_iter().flatten())
                    .map(|v| v.as_u64().unwrap() as u8)
                    .collect();
                let index: Vec<u32> = (case["grid"]["index"].as_array().into_iter().flatten())
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect();
                let p = packing(&case["packing"], &slots);
                let g = grid(&case["grid"], &index);
                let q = p.unpack(&b64(&case["words"]), rows, cols).unwrap();
                let s = f32s(&case["scale"]);
                if core == "table" {
                    table::decode(&q, &f32s(&case["table"]), &s, &g, rows, cols).unwrap()
                } else {
                    let (zero, bias) = (case.get("zero").map(f32s), case.get("bias").map(f32s));
                    let offset = match (&zero, &bias) {
                        (Some(z), _) => Offset::Zero(z),
                        (_, Some(b)) => Offset::Bias(b),
                        _ => Offset::None,
                    };
                    dequant(&q, &s, offset, &g, rows, cols)
                }
            }
        };
        let want = f32s(&case["expect"]);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&got), bits(&want), "{}", case["name"]);
        seen += 1;
    }
    assert_eq!(seen, 16);
}

/// Every fixed table in the spec's Appendix A matches its SHA-256: "`<dtype> [shape]`, SHA-256
/// `<hex>`:" followed by the values in a fenced block.
#[test]
fn spec_tables_match_their_digests() {
    use sha2::{Digest, Sha256};
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("spec/cozytensors.md");
    let spec = std::fs::read_to_string(path).unwrap();
    let appendix = &spec[spec.find("## Appendix A.").unwrap()..];
    let mut lines = appendix.lines();
    let mut tables = 0;
    while let Some(line) = lines.next() {
        let Some((head, rest)) = line.split_once("`, SHA-256 `") else {
            continue;
        };
        let (dtype, shape) = head.trim_start_matches('`').split_once(' ').unwrap();
        let want = rest.trim_end_matches("`:");
        assert_eq!(lines.next(), Some(""));
        assert_eq!(lines.next(), Some("```"));
        let text: String = lines
            .by_ref()
            .take_while(|l| *l != "```")
            .collect::<Vec<_>>()
            .join(" ");
        let values: Vec<&str> = text
            .split(|c: char| "[], ".contains(c))
            .filter(|v| !v.is_empty())
            .collect();
        let count: usize = (shape.trim_matches(|c| c == '[' || c == ']').split(", "))
            .map(|d| d.parse::<usize>().unwrap())
            .product();
        assert_eq!(values.len(), count, "{line}");
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| match dtype {
                "I8" => vec![v.parse::<i8>().unwrap() as u8],
                "U8" => vec![v.parse::<u8>().unwrap()],
                "I32" => v.parse::<i32>().unwrap().to_le_bytes().to_vec(),
                "F32" => v.parse::<f32>().unwrap().to_le_bytes().to_vec(),
                other => panic!("no table dtype {other}"),
            })
            .collect();
        let got: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(got, want, "{line}");
        tables += 1;
    }
    assert!(tables >= 1);
}
