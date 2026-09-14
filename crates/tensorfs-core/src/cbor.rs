//! The one stored CozyTensors-header codec: a closed all-array RFC 8949 Core
//! Deterministic CBOR profile. Generic CBOR values never enter the model.

use minicbor::data::Type;
use minicbor::{Decoder, Encoder};

use crate::dtype::Dtype;
use crate::err::{refuse, Code, Refusal, Result};
use crate::header::{Asset, Body, Header, Part, Tensor};
use crate::ids::ObjectRef;
use crate::limits;
use crate::relation::{Carrier, Dim, Relation};
use crate::spec::{EncodingSpec, Role};

fn malformed(what: &str, error: impl std::fmt::Display) -> Refusal {
    Refusal {
        code: Code::MALFORMED_CBOR,
        detail: format!("{what}: {error}"),
    }
}

struct Bounded {
    bytes: Vec<u8>,
}

#[derive(Debug)]
struct CapError;

impl std::fmt::Display for CapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("header byte cap exceeded")
    }
}

impl minicbor::encode::Write for Bounded {
    type Error = CapError;

    fn write_all(&mut self, bytes: &[u8]) -> std::result::Result<(), Self::Error> {
        let length = self.bytes.len().checked_add(bytes.len()).ok_or(CapError)?;
        if length > Header::MAX_BYTES {
            return Err(CapError);
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
}

fn emit<T>(
    what: &str,
    result: std::result::Result<T, minicbor::encode::Error<CapError>>,
) -> Result<T> {
    result.map_err(|error| {
        if error.is_write() {
            Refusal {
                code: Code::SIZE_CAP,
                detail: format!("{what}: encoded header exceeds {} bytes", Header::MAX_BYTES),
            }
        } else {
            malformed(what, error)
        }
    })
}

fn exact_array(d: &mut Decoder<'_>, what: &str, want: u64) -> Result<()> {
    match d.array().map_err(|e| malformed(what, e))? {
        Some(got) if got == want => Ok(()),
        Some(got) => refuse(
            Code::WRONG_TYPE,
            format!("{what}: array arity {got}, expected {want}"),
        ),
        None => refuse(
            Code::NONCANONICAL_ENCODING,
            format!("{what}: indefinite arrays are forbidden"),
        ),
    }
}

fn list_len(d: &mut Decoder<'_>, what: &str, cap: usize, nonempty: bool) -> Result<usize> {
    let n = match d.array().map_err(|e| malformed(what, e))? {
        Some(n) => usize::try_from(n).map_err(|_| malformed(what, "array length over usize"))?,
        None => {
            return refuse(
                Code::NONCANONICAL_ENCODING,
                format!("{what}: indefinite arrays are forbidden"),
            )
        }
    };
    if n > cap || (nonempty && n == 0) {
        return refuse(
            Code::COUNT_CAP,
            format!(
                "{what}: count {n} outside {}..={cap}",
                usize::from(nonempty)
            ),
        );
    }
    Ok(n)
}

fn text(d: &mut Decoder<'_>, what: &str, max: usize) -> Result<String> {
    let value = d.str().map_err(|e| malformed(what, e))?;
    if value.is_empty() || value.len() > max || !value.bytes().all(|b| (0x20..=0x7e).contains(&b)) {
        return refuse(
            Code::NON_ASCII_FIELD,
            format!("{what}: expected 1..={max} printable-ASCII bytes"),
        );
    }
    Ok(value.to_string())
}

fn uint(d: &mut Decoder<'_>, what: &str) -> Result<u64> {
    d.u64().map_err(|e| malformed(what, e))
}

fn digest_bytes(what: &str, hex: &str) -> Result<[u8; 32]> {
    crate::ids::hex64(what, hex)?;
    let mut out = [0u8; 32];
    for (i, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
        let nibble = |b: u8| match b {
            b'0'..=b'9' => b - b'0',
            b'a'..=b'f' => b - b'a' + 10,
            _ => 0,
        };
        out[i] = (nibble(pair[0]) << 4) | nibble(pair[1]);
    }
    Ok(out)
}

fn digest_hex(d: &mut Decoder<'_>, what: &str) -> Result<String> {
    let bytes = d.bytes().map_err(|e| malformed(what, e))?;
    if bytes.len() != 32 {
        return refuse(
            Code::MALFORMED_DIGEST,
            format!("{what}: digest is {} bytes, expected 32", bytes.len()),
        );
    }
    Ok(crate::sha256::hex(bytes))
}

fn encode_ref(e: &mut Encoder<Bounded>, r: &ObjectRef) -> Result<()> {
    emit("ObjectRef", e.array(2))?;
    emit(
        "ObjectRef.digest",
        e.bytes(&digest_bytes("ObjectRef", &r.sha256)?),
    )?;
    emit("ObjectRef.length", e.u64(r.length))?;
    Ok(())
}

fn decode_ref(d: &mut Decoder<'_>, what: &str) -> Result<ObjectRef> {
    exact_array(d, what, 2)?;
    Ok(ObjectRef {
        sha256: digest_hex(d, what)?,
        length: uint(d, what)?,
    })
}

fn encode_shape(e: &mut Encoder<Bounded>, shape: &[u64]) -> Result<()> {
    emit("shape", e.array(shape.len() as u64))?;
    for extent in shape {
        emit("shape.extent", e.u64(*extent))?;
    }
    Ok(())
}

fn decode_shape(d: &mut Decoder<'_>, what: &str) -> Result<Vec<u64>> {
    let n = list_len(d, what, limits::MAX_RANK, false)?;
    (0..n).map(|_| uint(d, what)).collect()
}

fn encode_optional<T>(
    e: &mut Encoder<Bounded>,
    value: Option<&T>,
    write: impl FnOnce(&mut Encoder<Bounded>, &T) -> Result<()>,
) -> Result<()> {
    emit("optional", e.array(u64::from(value.is_some())))?;
    if let Some(value) = value {
        write(e, value)?;
    }
    Ok(())
}

fn decode_optional<T>(
    d: &mut Decoder<'_>,
    what: &str,
    read: impl FnOnce(&mut Decoder<'_>) -> Result<T>,
) -> Result<Option<T>> {
    match list_len(d, what, 1, false)? {
        0 => Ok(None),
        1 => read(d).map(Some),
        _ => unreachable!(),
    }
}

fn encode_carrier(e: &mut Encoder<Bounded>, carrier: &Carrier) -> Result<()> {
    match carrier {
        Carrier::SameAsLogical => {
            emit("encoding.carrier", e.array(1))?;
            emit("encoding.carrier.tag", e.u8(0))?;
        }
        Carrier::Set(dtypes) => {
            emit("encoding.carrier", e.array(2))?;
            emit("encoding.carrier.tag", e.u8(1))?;
            emit("encoding.carrier.dtypes", e.array(dtypes.len() as u64))?;
            for dtype in dtypes {
                emit("encoding.carrier.dtype", e.u8(dtype.code()))?;
            }
        }
    }
    Ok(())
}

fn decode_carrier(d: &mut Decoder<'_>, what: &str) -> Result<Carrier> {
    let length = list_len(d, what, 2, true)?;
    match uint(d, &format!("{what}.tag"))? {
        0 if length == 1 => Ok(Carrier::SameAsLogical),
        1 if length == 2 => {
            let count = list_len(d, &format!("{what}.dtypes"), 32, true)?;
            let dtypes = (0..count)
                .map(|_| Dtype::from_code(uint(d, &format!("{what}.dtype"))?))
                .collect::<Result<Vec<_>>>()?;
            Ok(Carrier::Set(dtypes))
        }
        tag => refuse(
            Code::UNKNOWN_RELATION,
            format!("{what}: carrier tag/arity ({tag},{length}) is unknown"),
        ),
    }
}

fn encode_dim(e: &mut Encoder<Bounded>, dim: &Dim) -> Result<()> {
    match dim {
        Dim::Axis { axis } => {
            emit("encoding.dim", e.array(2))?;
            emit("encoding.dim.tag", e.u8(0))?;
            emit("encoding.dim.axis", e.u64(*axis))?;
        }
        Dim::Div { axis, by } => {
            emit("encoding.dim", e.array(3))?;
            emit("encoding.dim.tag", e.u8(1))?;
            emit("encoding.dim.axis", e.u64(*axis))?;
            emit("encoding.dim.by", e.u64(*by))?;
        }
        Dim::CeilDiv { axis, by } => {
            emit("encoding.dim", e.array(3))?;
            emit("encoding.dim.tag", e.u8(2))?;
            emit("encoding.dim.axis", e.u64(*axis))?;
            emit("encoding.dim.by", e.u64(*by))?;
        }
        Dim::CeilBlock { axis, block, mul } => {
            emit("encoding.dim", e.array(4))?;
            emit("encoding.dim.tag", e.u8(3))?;
            emit("encoding.dim.axis", e.u64(*axis))?;
            emit("encoding.dim.block", e.u64(*block))?;
            emit("encoding.dim.mul", e.u64(*mul))?;
        }
        Dim::ElementsDiv { by } => {
            emit("encoding.dim", e.array(2))?;
            emit("encoding.dim.tag", e.u8(4))?;
            emit("encoding.dim.by", e.u64(*by))?;
        }
        Dim::Lit { value } => {
            emit("encoding.dim", e.array(2))?;
            emit("encoding.dim.tag", e.u8(5))?;
            emit("encoding.dim.value", e.u64(*value))?;
        }
    }
    Ok(())
}

fn decode_dim(d: &mut Decoder<'_>, what: &str) -> Result<Dim> {
    let length = list_len(d, what, 4, true)?;
    let tag = uint(d, &format!("{what}.tag"))?;
    let dim = match (tag, length) {
        (0, 2) => Dim::Axis {
            axis: uint(d, &format!("{what}.axis"))?,
        },
        (1, 3) => Dim::Div {
            axis: uint(d, &format!("{what}.axis"))?,
            by: uint(d, &format!("{what}.by"))?,
        },
        (2, 3) => Dim::CeilDiv {
            axis: uint(d, &format!("{what}.axis"))?,
            by: uint(d, &format!("{what}.by"))?,
        },
        (3, 4) => Dim::CeilBlock {
            axis: uint(d, &format!("{what}.axis"))?,
            block: uint(d, &format!("{what}.block"))?,
            mul: uint(d, &format!("{what}.mul"))?,
        },
        (4, 2) => Dim::ElementsDiv {
            by: uint(d, &format!("{what}.by"))?,
        },
        (5, 2) => Dim::Lit {
            value: uint(d, &format!("{what}.value"))?,
        },
        _ => {
            return refuse(
                Code::UNKNOWN_RELATION,
                format!("{what}: dimension tag/arity ({tag},{length}) is unknown"),
            )
        }
    };
    Ok(dim)
}

fn encode_relation(e: &mut Encoder<Bounded>, relation: &Relation) -> Result<()> {
    match relation {
        Relation::Same => {
            emit("encoding.relation", e.array(1))?;
            emit("encoding.relation.tag", e.u8(0))?;
        }
        Relation::Dims(dims) => {
            emit("encoding.relation", e.array(2))?;
            emit("encoding.relation.tag", e.u8(1))?;
            emit("encoding.relation.dims", e.array(dims.len() as u64))?;
            for dim in dims {
                encode_dim(e, dim)?;
            }
        }
    }
    Ok(())
}

fn decode_relation(d: &mut Decoder<'_>, what: &str) -> Result<Relation> {
    let length = list_len(d, what, 2, true)?;
    match uint(d, &format!("{what}.tag"))? {
        0 if length == 1 => Ok(Relation::Same),
        1 if length == 2 => {
            let count = list_len(d, &format!("{what}.dims"), limits::MAX_RANK, false)?;
            let dims = (0..count)
                .map(|index| decode_dim(d, &format!("{what}.dims[{index}]")))
                .collect::<Result<Vec<_>>>()?;
            Ok(Relation::Dims(dims))
        }
        tag => refuse(
            Code::UNKNOWN_RELATION,
            format!("{what}: relation tag/arity ({tag},{length}) is unknown"),
        ),
    }
}

fn encode_encoding_into(e: &mut Encoder<Bounded>, spec: &EncodingSpec) -> Result<()> {
    emit("encoding", e.array(4))?;
    emit(
        "encoding.logical_dtypes",
        e.array(spec.logical_dtypes.len() as u64),
    )?;
    for dtype in &spec.logical_dtypes {
        emit("encoding.logical_dtype", e.u8(dtype.code()))?;
    }
    encode_optional(e, spec.logical_rank.as_ref(), |e, rank| {
        emit("encoding.logical_rank", e.u64(*rank))?;
        Ok(())
    })?;
    emit("encoding.roles", e.array(spec.roles.len() as u64))?;
    for (name, role) in &spec.roles {
        emit("encoding.role", e.array(3))?;
        emit("encoding.role.name", e.str(name))?;
        encode_carrier(e, &role.carrier)?;
        encode_relation(e, &role.shape)?;
    }
    encode_optional(e, spec.vectors.as_ref(), encode_ref)?;
    Ok(())
}

fn decode_encoding_from(d: &mut Decoder<'_>, what: &str) -> Result<EncodingSpec> {
    exact_array(d, what, 4)?;
    let count = list_len(d, &format!("{what}.logical_dtypes"), 32, true)?;
    let logical_dtypes = (0..count)
        .map(|_| Dtype::from_code(uint(d, &format!("{what}.logical_dtype"))?))
        .collect::<Result<Vec<_>>>()?;
    let logical_rank = decode_optional(d, &format!("{what}.logical_rank"), |d| {
        uint(d, &format!("{what}.logical_rank.value"))
    })?;
    let count = list_len(d, &format!("{what}.roles"), limits::MAX_PARTS, true)?;
    let mut roles = Vec::with_capacity(count);
    for index in 0..count {
        let role_what = format!("{what}.roles[{index}]");
        exact_array(d, &role_what, 3)?;
        let name = text(d, &format!("{role_what}.name"), limits::MAX_NAME_BYTES)?;
        let carrier = decode_carrier(d, &format!("{role_what}.carrier"))?;
        let shape = decode_relation(d, &format!("{role_what}.shape"))?;
        roles.push((name, Role { carrier, shape }));
    }
    let vectors = decode_optional(d, &format!("{what}.vectors"), |d| {
        decode_ref(d, &format!("{what}.vectors.value"))
    })?;
    let spec = EncodingSpec {
        logical_dtypes,
        logical_rank,
        roles,
        vectors,
    };
    EncodingSpec::from_value(&spec.to_value())
}

pub(crate) fn encode_encoding(spec: &EncodingSpec) -> Result<Vec<u8>> {
    EncodingSpec::from_value(&spec.to_value())?;
    let mut encoder = Encoder::new(Bounded { bytes: Vec::new() });
    encode_encoding_into(&mut encoder, spec)?;
    Ok(encoder.into_writer().bytes)
}

pub(crate) fn decode_encoding(bytes: &[u8]) -> Result<EncodingSpec> {
    if bytes.len() > limits::SPEC_MAX_BYTES {
        return refuse(Code::SIZE_CAP, "nested encoding exceeds its cap");
    }
    let mut decoder = Decoder::new(bytes);
    let spec = decode_encoding_from(&mut decoder, "encoding")?;
    if decoder.position() != bytes.len() || encode_encoding(&spec)? != bytes {
        return refuse(
            Code::NONCANONICAL_ENCODING,
            "encoding bytes are not the exact nested deterministic CBOR value",
        );
    }
    Ok(spec)
}

fn encode_part(e: &mut Encoder<Bounded>, role: &str, part: &Part) -> Result<()> {
    emit("part", e.array(4))?;
    emit("part.role", e.str(role))?;
    emit("part.dtype", e.u8(part.dtype.code()))?;
    encode_shape(e, &part.shape)?;
    match &part.body {
        Body::Inline(bytes) => {
            emit("part.inline", e.bytes(bytes))?;
        }
        Body::Segments(segments) => {
            emit("part.segments", e.array(segments.len() as u64))?;
            for segment in segments {
                encode_ref(e, segment)?;
            }
        }
    }
    Ok(())
}

pub(crate) fn encoded_tensor_identity(tensor: &Tensor) -> Result<String> {
    let mut encoder = Encoder::new(Bounded { bytes: Vec::new() });
    emit("tensor identity", encoder.array(5))?;
    emit(
        "tensor identity domain",
        encoder.str("tensorfs.encoded-tensor/1"),
    )?;
    emit("tensor identity dtype", encoder.u8(tensor.dtype.code()))?;
    encode_shape(&mut encoder, &tensor.shape)?;
    emit("tensor identity encoding", encoder.str(&tensor.encoding))?;
    emit(
        "tensor identity parts",
        encoder.array(tensor.parts.len() as u64),
    )?;
    for (role, part) in &tensor.parts {
        encode_part(&mut encoder, role, part)?;
    }
    Ok(crate::ids::object_id(&encoder.into_writer().bytes))
}

fn decode_part(d: &mut Decoder<'_>, what: &str) -> Result<(String, Part)> {
    exact_array(d, what, 4)?;
    let role = text(d, &format!("{what}.role"), limits::MAX_NAME_BYTES)?;
    let dtype = Dtype::from_code(uint(d, &format!("{what}.dtype"))?)?;
    let shape = decode_shape(d, &format!("{what}.shape"))?;
    let body = match d.datatype().map_err(|e| malformed(what, e))? {
        Type::Bytes => {
            let bytes = d.bytes().map_err(|e| malformed(what, e))?;
            if bytes.len() as u64 > limits::INLINE_MAX_BYTES {
                return refuse(
                    Code::INLINE_THRESHOLD,
                    format!(
                        "{what}.body: {} inline bytes exceeds {}",
                        bytes.len(),
                        limits::INLINE_MAX_BYTES
                    ),
                );
            }
            Body::Inline(bytes.to_vec())
        }
        Type::Array => {
            let n = list_len(d, &format!("{what}.segments"), limits::MAX_SEGMENTS, true)?;
            let mut segments = Vec::with_capacity(n);
            for i in 0..n {
                segments.push(decode_ref(d, &format!("{what}.segments[{i}]"))?);
            }
            Body::Segments(segments)
        }
        got => {
            return refuse(
                Code::WRONG_TYPE,
                format!("{what}.body: expected bytes or array, got {got}"),
            )
        }
    };
    Ok((role, Part { dtype, shape, body }))
}

fn encode_tensor(
    e: &mut Encoder<Bounded>,
    key: &str,
    tensor: &Tensor,
    encodings: &std::collections::HashMap<String, usize>,
) -> Result<()> {
    emit("tensor", e.array(5))?;
    emit("tensor.key", e.str(key))?;
    emit("tensor.dtype", e.u8(tensor.dtype.code()))?;
    encode_shape(e, &tensor.shape)?;
    let index = encodings
        .get(&tensor.encoding)
        .copied()
        .ok_or_else(|| Refusal {
            code: Code::ENCODING_MISMATCH,
            detail: format!("{key}: encoding {} is not listed", tensor.encoding),
        })?;
    emit("tensor.encoding", e.u64(index as u64))?;
    emit("tensor.parts", e.array(tensor.parts.len() as u64))?;
    for (role, part) in &tensor.parts {
        encode_part(e, role, part)?;
    }
    Ok(())
}

fn decode_tensor(
    d: &mut Decoder<'_>,
    what: &str,
    encodings: &[EncodingSpec],
) -> Result<(String, Tensor)> {
    exact_array(d, what, 5)?;
    let key = text(d, &format!("{what}.key"), limits::MAX_KEY_BYTES)?;
    let dtype = Dtype::from_code(uint(d, &format!("{what}.dtype"))?)?;
    let shape = decode_shape(d, &format!("{what}.shape"))?;
    let encoding_index =
        usize::try_from(uint(d, &format!("{what}.encoding"))?).map_err(|_| Refusal {
            code: Code::NUMBER_RANGE,
            detail: format!("{what}: encoding index is outside this platform's usize range"),
        })?;
    let encoding = encodings
        .get(encoding_index)
        .ok_or_else(|| Refusal {
            code: Code::ENCODING_MISMATCH,
            detail: format!(
                "{what}: encoding index {encoding_index} outside {} entries",
                encodings.len()
            ),
        })?
        .object_id();
    let n = list_len(d, &format!("{what}.parts"), limits::MAX_PARTS, true)?;
    let mut parts = Vec::with_capacity(n);
    for i in 0..n {
        parts.push(decode_part(d, &format!("{what}.parts[{i}]"))?);
    }
    Ok((
        key,
        Tensor {
            dtype,
            shape,
            encoding,
            parts,
        },
    ))
}

pub fn encode_header(header: &Header) -> Result<Vec<u8>> {
    header.validate_structure()?;
    encode_header_unvalidated(header)
}

pub fn encode_header_unvalidated(header: &Header) -> Result<Vec<u8>> {
    let mut e = Encoder::new(Bounded { bytes: Vec::new() });
    let encoding_index: std::collections::HashMap<String, usize> = header
        .encodings
        .iter()
        .enumerate()
        .map(|(index, encoding)| (encoding.object_id(), index))
        .collect();
    emit("header", e.array(5))?;
    emit("header.format", e.str(Header::FORMAT))?;

    emit("header.configs", e.array(header.configs.len() as u64))?;
    for (name, config) in &header.configs {
        emit("config", e.array(2))?;
        emit("config.name", e.str(name))?;
        emit("config.canonical_json", e.bytes(config))?;
    }

    emit("header.assets", e.array(header.assets.len() as u64))?;
    for (name, asset) in &header.assets {
        emit("asset", e.array(5))?;
        emit("asset.name", e.str(name))?;
        emit(
            "asset.logical_digest",
            e.bytes(&digest_bytes(
                "asset logical digest",
                &asset.logical_sha256,
            )?),
        )?;
        emit("asset.logical_length", e.u64(asset.logical_length))?;
        emit("asset.media_type", e.str(&asset.media_type))?;
        emit("asset.segments", e.array(asset.segments.len() as u64))?;
        for segment in &asset.segments {
            encode_ref(&mut e, segment)?;
        }
    }

    emit("header.encodings", e.array(header.encodings.len() as u64))?;
    for encoding in &header.encodings {
        encode_encoding_into(&mut e, encoding)?;
    }

    emit("header.components", e.array(header.components.len() as u64))?;
    for (name, tensors) in &header.components {
        emit("component", e.array(2))?;
        emit("component.name", e.str(name))?;
        emit("component.tensors", e.array(tensors.len() as u64))?;
        for (key, tensor) in tensors {
            encode_tensor(&mut e, key, tensor, &encoding_index)?;
        }
    }
    Ok(e.into_writer().bytes)
}

fn decode(bytes: &[u8]) -> Result<Header> {
    if bytes.len() > Header::MAX_BYTES {
        return refuse(
            Code::SIZE_CAP,
            format!(
                "header is {} bytes, cap is {}",
                bytes.len(),
                Header::MAX_BYTES
            ),
        );
    }
    let mut d = Decoder::new(bytes);
    exact_array(&mut d, "header", 5)?;
    let format = text(&mut d, "header.format", 64)?;
    if format != Header::FORMAT {
        return refuse(
            Code::UNKNOWN_FORMAT,
            format!("Header: format {format:?} != {:?}", Header::FORMAT),
        );
    }

    let config_count = list_len(&mut d, "header.configs", limits::MAX_COMPONENTS, false)?;
    let mut configs = Vec::with_capacity(config_count);
    for i in 0..config_count {
        let what = format!("header.configs[{i}]");
        exact_array(&mut d, &what, 2)?;
        let name = text(&mut d, &format!("{what}.name"), limits::MAX_NAME_BYTES)?;
        let config = d
            .bytes()
            .map_err(|error| malformed(&format!("{what}.canonical_json"), error))?;
        if config.len() > limits::DOC_MAX_BYTES {
            return refuse(
                Code::SIZE_CAP,
                format!("{what}.canonical_json exceeds the document byte cap"),
            );
        }
        configs.push((name, config.to_vec()));
    }

    let asset_count = list_len(&mut d, "header.assets", limits::MAX_ENTRIES, false)?;
    let mut assets = Vec::with_capacity(asset_count);
    for i in 0..asset_count {
        let what = format!("header.assets[{i}]");
        exact_array(&mut d, &what, 5)?;
        let name = text(&mut d, &format!("{what}.name"), limits::MAX_KEY_BYTES)?;
        let logical_sha256 = digest_hex(&mut d, &format!("{what}.logical_digest"))?;
        let logical_length = uint(&mut d, &format!("{what}.logical_length"))?;
        let media_type = text(&mut d, &format!("{what}.media_type"), 128)?;
        let count = list_len(
            &mut d,
            &format!("{what}.segments"),
            limits::MAX_SEGMENTS,
            true,
        )?;
        let mut segments = Vec::with_capacity(count);
        for j in 0..count {
            segments.push(decode_ref(&mut d, &format!("{what}.segments[{j}]"))?);
        }
        assets.push((
            name,
            Asset {
                logical_sha256,
                logical_length,
                media_type,
                segments,
            },
        ));
    }

    let encoding_count = list_len(&mut d, "header.encodings", limits::MAX_TOTAL_REFS, true)?;
    let mut encodings = Vec::with_capacity(encoding_count);
    for i in 0..encoding_count {
        let what = format!("header.encodings[{i}]");
        encodings.push(decode_encoding_from(&mut d, &what)?);
    }

    let component_count = list_len(&mut d, "header.components", limits::MAX_COMPONENTS, false)?;
    let mut components = Vec::with_capacity(component_count);
    let mut tensor_count = 0usize;
    for i in 0..component_count {
        let what = format!("header.components[{i}]");
        exact_array(&mut d, &what, 2)?;
        let name = text(&mut d, &format!("{what}.name"), limits::MAX_NAME_BYTES)?;
        let count = list_len(
            &mut d,
            &format!("{what}.tensors"),
            limits::MAX_TENSORS,
            true,
        )?;
        tensor_count = tensor_count
            .checked_add(count)
            .ok_or_else(|| malformed(&what, "tensor count overflow"))?;
        if tensor_count > limits::MAX_TENSORS {
            return refuse(Code::COUNT_CAP, "whole-header tensor count over cap");
        }
        let mut tensors = Vec::with_capacity(count);
        for j in 0..count {
            tensors.push(decode_tensor(
                &mut d,
                &format!("{what}.tensors[{j}]"),
                &encodings,
            )?);
        }
        components.push((name, tensors));
    }
    if d.position() != bytes.len() {
        return refuse(
            Code::TRAILING_BYTES,
            format!("header: trailing bytes at offset {}", d.position()),
        );
    }
    Ok(Header {
        configs,
        assets,
        encodings,
        components,
    })
}

pub fn decode_header(bytes: &[u8]) -> Result<Header> {
    let header = decode(bytes)?;
    if encode_header(&header)? != bytes {
        return refuse(
            Code::NONCANONICAL_ENCODING,
            "stored header bytes are not the restricted deterministic CBOR encoding of their content",
        );
    }
    header.validate_structure()?;
    Ok(header)
}
