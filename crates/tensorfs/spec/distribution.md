# cozytensors distribution/1

Status: normative target for TensorFS 0.13.0; implementation is being integrated.

How a client reads and publishes cozytensors repos at an **endpoint**, so any server, bucket or
directory can serve them. Tensorhub implements it; `tfs conform endpoint` checks an implementation. Documents and
objects are as in [cozytensors.md](cozytensors.md). The key words are as in RFC 2119.

## 1. Resources

An endpoint is a base URL `B`. It has two resources, and nothing lists either:

```text
B/refs/<org>/<name>.json        the repo document (tensorfs.repo/1)
B/blobs/<aa>/<bb>/<sha256>      an object: a tree, a cozytensors header, a chunk, an asset or a repo file
```

- `<org>/<name>` is the repo name with ASCII `A`–`Z` lowered; `<aa>` and `<bb>` are the first two pairs of the hex
  digest. A store's directory has the same layout, so a directory, bucket or static host serves it as it is.
- URL construction percent-encodes each UTF-8 name segment separately after ASCII case folding; `/` remains
  the separator. A name containing a space, `%`, `?` or `#` must not alter the URL structure
  ([RFC 3986 §3.3](https://www.rfc-editor.org/rfc/rfc3986.html#section-3.3)).
- The org `local` is reserved for a machine's own repos and is never served.

## 2. Reading

```text
GET  B/refs/<org>/<name>.json   [If-None-Match: <etag>]     200 + ETag | 304 | 404
GET  B/blobs/<aa>/<bb>/<hex>    [Range: bytes=a-b]          200 | 206 | 302/303/307 + Location | 404
HEAD B/blobs/<aa>/<bb>/<hex>                                200 + Content-Length | 404
```

- A client walks repo → tree → its repo files and the chosen variant's header → that header's chunks and assets. It
  MUST check every object's length and SHA-256 before it admits it.
- A read-only server SHOULD send an `ETag` with a repo document and answer `304` to a matching `If-None-Match`.
  A server that accepts repo updates MUST send a strong ETag on successful repo GET and PUT responses. Weak
  validators cannot satisfy the strong comparison required by `If-Match`
  ([RFC 9110 §13.1.1](https://www.rfc-editor.org/rfc/rfc9110.html#name-if-match)).
- A server SHOULD honour one `bytes=a-b` range on a blob. A client MUST accept a `200` with the whole object instead.
- A blob GET MAY redirect. The client repeats the request at `Location` with its `Range`, follows at most 5 hops, and
  MAY reuse the target for restarts until the redirect's `Cache-Control: max-age` ends. Plain `http` is allowed only
  on loopback.

## 3. Uploading objects

For each object the client sends HEAD, then PUT when it is absent, then HEAD:

```text
HEAD B/blobs/<aa>/<bb>/<hex>            200 (present) | 404 [+ Location: where to PUT it]
PUT  <that Location, else the blob URL>
     Content-Length: <length>
     x-amz-checksum-sha256: <base64 of the digest>
     If-None-Match: *                   200 | 201 | 412 (present) | 400 digest_mismatch | 413 too_large
HEAD B/blobs/<aa>/<bb>/<hex>            200: the object is confirmed
```

- Order: chunks, assets and repo files first, then headers, then the tree. A tree present then implies its closure was
  sent.
- A server MUST store an object only when its bytes match the digest. `412` means it is already there, which is
  success: objects are content-addressed.
- The PUT always carries the three headers, so a presigned URL that signed them takes the same request. A server
  that redirects uploads verifies the object at the confirming HEAD.

## 4. Publishing

```text
PUT  B/refs/<org>/<name>.json   If-Match: <etag> | If-None-Match: *
     200/201 + ETag | 412 | 428 | 409 closure_incomplete | 409 policy | 400 invalid
```

- `If-None-Match: *` creates the document; `If-Match` replaces the version the client read. Without either a server
  answers `428`.
- On `412` the client reads the document again, applies its change to it and retries. A retry after a lost answer
  that gets `412` reads it again: when it equals what was sent, the write succeeded.
- A model is deleted by a document with no releases and no checkpoints. There is no DELETE.
- A server checks:
  - the body is a canonical `tensorfs.repo/1` document (cozytensors.md §9) whose `name` folds to the path;
  - every tree it cites has its closure present and readable by the writer, else `409 closure_incomplete` with up to
    1,000 missing digests in `detail.missing`;
  - a release that changed has `revision` = old + 1, an unchanged one keeps it, and a new one starts at 0;
  - anything else is the server's policy: `409 policy`, `detail.reason` naming the server's own code.

## 5. Authorization

- Reads of public objects and documents are anonymous. Writes and private reads carry `Authorization`: `Bearer`,
  `DPoP` (RFC 9449, with `DPoP-Nonce` retries) or S3 SigV4.
- `401` carries `WWW-Authenticate`. Without read access an object or repo is `404`, never `403`.
- A client MUST NOT send credentials to another origin, after a redirect or a `Location`.

## 6. Errors

- An error body is `{"error": {"code", "message", "detail"?}}`.
- Codes: `not_found` 404, `unauthorized` 401, `denied` 403, `invalid` 400, `digest_mismatch` 400,
  `precondition_failed` 412, `precondition_required` 428, `closure_incomplete` 409, `policy` 409, `too_large` 413
  (`detail.limit`), `rate_limited` 429, `unavailable` 503, `unsupported` 405 (a read-only endpoint).
- A static server sends bare statuses; a client maps them by number.
- A client retries `429` and `503` at `Retry-After`, and transport failures while they make progress. It retries no
  other `4xx`, except the `412` loop and one `401` after its credential provider refreshes.

## 7. Limits and versioning

- An object is at most 5 GiB (chunks are at most 64 MiB) and a repo document at most 16 MiB. A server MAY set lower
  limits and states them in `413` `detail.limit`.
- This is distribution/1. Documents carry their own format tags. New optional headers, codes and fields are minor and
  ignored by those who don't know them. A breaking change takes a new key root (`v2/…`). There is no handshake.

## 8. Endpoints without a server

| Endpoint | Reads | Writes |
|---|---|---|
| A directory, `file:///dir` | the files | objects by temp and rename, verified; refs under the store's record lock, compared before replace |
| A static host (nginx, Caddy, `python -m http.server`) | as is | none (`405`) |
| S3 or R2, `s3://bucket/prefix` | public, or SigV4 | SigV4: objects with `If-None-Match: *` and the checksum (the bucket verifies); refs with `If-Match` / `If-None-Match` |

A bucket has no closure check; the upload order keeps closures whole.
