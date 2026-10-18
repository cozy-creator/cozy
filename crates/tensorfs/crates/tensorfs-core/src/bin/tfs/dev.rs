//! `tfs dev ckpt`: a synthetic SDXL-shaped checkpoint written through the v2 writer, for
//! Tensorhub's tests and ours. Bytes are a splitmix64 stream keyed by each part's name, so a
//! reader can regenerate them. Unless `--plain`, attention and MLP weights are encoded with
//! every scheme (finite scales); one tensor spans two 64 MiB grid chunks at scale 1, and two
//! tensors share their bytes.

use std::io::Read;

use tensorfs_core::failure::Result;
use tensorfs_core::format::{Dtype, Encoding, Metadata, ObjectRef, Value};
use tensorfs_core::store2::Store;
use tensorfs_core::writer::{Data, PartData, Source};

pub struct Noise {
    state: u64,
    left: u64,
}

impl Noise {
    pub fn new(seed: &str, len: u64) -> Noise {
        // FNV-1a of the seed.
        let state = seed.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
        });
        Noise { state, left: len }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn bytes(seed: &str, len: u64) -> Vec<u8> {
        let mut out = Vec::with_capacity(len as usize);
        Noise::new(seed, len)
            .read_to_end(&mut out)
            .expect("noise reads");
        out
    }
}

impl Read for Noise {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = (buf.len() as u64).min(self.left) as usize;
        for chunk in buf[..n].chunks_mut(8) {
            let word = self.next().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
        self.left -= n as u64;
        Ok(n)
    }
}

/// E4M3 bytes with the NaN codes moved one step down.
fn fp8(seed: &str, n: u64) -> Vec<u8> {
    let mut b = Noise::bytes(seed, n);
    b.iter_mut()
        .filter(|v| **v & 0x7f == 0x7f)
        .for_each(|v| *v ^= 1);
    b
}

/// `n` finite F32 scales near 1/64.
fn scales(seed: &str, n: u64) -> Vec<u8> {
    Noise::bytes(seed, n * 2)
        .chunks(2)
        .flat_map(|c| {
            ((1.0 + u16::from_le_bytes([c[0], c[1]]) as f32 / 65536.0) / 64.0).to_le_bytes()
        })
        .collect()
}

enum Scheme {
    Plain,
    Fp8Rowwise,
    Int8Convrot,
    Nvfp4,
}

struct Item {
    component: &'static str,
    key: String,
    dtype: Dtype,
    shape: Vec<u64>,
    scheme: Scheme,
    seed: String,
}

fn plan(blocks: u64, scale: u64, plain: bool) -> Vec<Item> {
    let d = |x: u64| (x / scale).max(16) / 16 * 16;
    let item = |component, key: String, dtype, shape: Vec<u64>, scheme| Item {
        seed: format!("{component}/{key}"),
        component,
        key,
        dtype,
        shape,
        scheme: if plain { Scheme::Plain } else { scheme },
    };
    let mut v = Vec::new();
    for i in 0..blocks {
        let k = |s: &str| format!("down_blocks.{i}.{s}");
        let square = vec![d(2048), d(2048)];
        let convrot = if d(2048) % 256 == 0 {
            Scheme::Int8Convrot
        } else {
            Scheme::Fp8Rowwise
        };
        v.push(item(
            "unet",
            k("attn1.to_q.weight"),
            Dtype::BF16,
            square.clone(),
            Scheme::Fp8Rowwise,
        ));
        v.push(item(
            "unet",
            k("attn1.to_k.weight"),
            Dtype::BF16,
            square.clone(),
            convrot,
        ));
        v.push(item(
            "unet",
            k("attn1.to_out.0.weight"),
            Dtype::BF16,
            square,
            Scheme::Plain,
        ));
        v.push(item(
            "unet",
            k("attn1.norm_q.weight"),
            Dtype::BF16,
            vec![d(64)],
            Scheme::Plain,
        ));
        v.push(item(
            "unet",
            k("ff.net.0.proj.weight"),
            Dtype::BF16,
            vec![d(2048), d(8192)],
            Scheme::Nvfp4,
        ));
        v.push(item(
            "unet",
            k("norm1.weight"),
            Dtype::BF16,
            vec![d(2048)],
            Scheme::Plain,
        ));
    }
    let embeddings = vec![d(8192), d(8192)];
    v.push(item(
        "text_encoder",
        "text_model.embeddings.token_embedding.weight".into(),
        Dtype::BF16,
        embeddings,
        Scheme::Plain,
    ));
    v.push(item(
        "text_encoder",
        "text_model.final_layer_norm.weight".into(),
        Dtype::F32,
        vec![d(1024)],
        Scheme::Plain,
    ));
    v.push(item(
        "vae",
        "decoder.conv_in.weight".into(),
        Dtype::F32,
        vec![d(512), d(512)],
        Scheme::Plain,
    ));
    let mut twin = item(
        "vae",
        "decoder.conv_out.weight".into(),
        Dtype::F32,
        vec![d(512), d(512)],
        Scheme::Plain,
    );
    twin.seed = "vae/decoder.conv_in.weight".into();
    v.push(twin);
    v
}

const FILES: [(&str, &str); 3] = [
    (
        "unet/config.json",
        r#"{"_class_name":"UNet2DConditionModel","block_out_channels":[320,640,1280],"in_channels":4}"#,
    ),
    (
        "text_encoder/config.json",
        r#"{"architectures":["CLIPTextModel"],"hidden_size":2048}"#,
    ),
    ("tokenizer/vocab.txt", "<pad>0</pad>\n<pad>1</pad>\n"),
];

/// Writes the checkpoint; its manifest, pinned under its digest.
pub fn ckpt(store: &Store, blocks: u64, scale: u64, plain: bool, files: bool) -> Result<ObjectRef> {
    let mut w = store.writer(None, None)?;
    for it in plan(blocks, scale, plain) {
        let (rows, cols) = (it.shape[0], it.shape.get(1).copied().unwrap_or(1));
        let seed = |role: &str| format!("{}#{role}", it.seed);
        let numel = rows * cols;
        let part = |role: &str, dtype: Dtype, shape: Vec<u64>, bytes: Vec<u8>| {
            (role.to_owned(), dtype, shape, bytes)
        };
        let (scheme, params, parts): (&str, Vec<(&str, Value)>, Vec<_>) = match it.scheme {
            Scheme::Plain => {
                let len = it
                    .dtype
                    .byte_len(&it.shape)
                    .expect("plan sizes are whole bytes");
                let mut noise = Noise::new(&seed(""), len);
                w.put(
                    it.component,
                    &it.key,
                    &it.dtype,
                    &it.shape,
                    Data::Plain(Source::Read(&mut noise)),
                )?;
                continue;
            }
            Scheme::Fp8Rowwise => (
                "fp8-rowwise",
                vec![("axis", 0u64.into())],
                vec![
                    part(
                        "weight",
                        Dtype::F8_E4M3,
                        it.shape.clone(),
                        fp8(&seed("weight"), numel),
                    ),
                    part(
                        "scale",
                        Dtype::F32,
                        vec![rows],
                        scales(&seed("scale"), rows),
                    ),
                ],
            ),
            Scheme::Int8Convrot => (
                "int8-convrot",
                vec![("block", 256u64.into()), ("rotation", "hadamard".into())],
                vec![
                    part(
                        "weight",
                        Dtype::I8,
                        it.shape.clone(),
                        Noise::bytes(&seed("weight"), numel),
                    ),
                    part(
                        "scale",
                        Dtype::F32,
                        vec![rows],
                        scales(&seed("scale"), rows),
                    ),
                ],
            ),
            Scheme::Nvfp4 => (
                "nvfp4",
                vec![("block", 16u64.into())],
                vec![
                    part(
                        "weight",
                        Dtype::F4,
                        it.shape.clone(),
                        Noise::bytes(&seed("weight"), numel / 2),
                    ),
                    part(
                        "scale",
                        Dtype::F8_E4M3,
                        vec![rows, cols / 16],
                        fp8(&seed("scale"), numel / 16),
                    ),
                    part("scale_2", Dtype::F32, vec![], scales(&seed("scale_2"), 1)),
                ],
            ),
        };
        let encoding = Encoding {
            scheme: scheme.into(),
            params: params
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v))
                .collect::<Metadata>(),
        };
        let mut data: Vec<(String, Dtype, Vec<u64>, &[u8])> = parts
            .iter()
            .map(|(r, d, s, b)| (r.clone(), d.clone(), s.clone(), b.as_slice()))
            .collect();
        let parts = data
            .iter_mut()
            .map(|(role, dtype, shape, bytes)| PartData {
                role: role.clone(),
                dtype: dtype.clone(),
                shape: shape.clone(),
                data: Source::Read(bytes),
            })
            .collect();
        w.put(
            it.component,
            &it.key,
            &it.dtype,
            &it.shape,
            Data::Encoded { encoding, parts },
        )?;
    }
    if files {
        for (path, text) in FILES {
            w.file(path, text.as_bytes())?;
        }
    }
    Ok(w.commit()?)
}
