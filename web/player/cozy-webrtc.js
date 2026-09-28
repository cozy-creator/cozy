// The reference browser client for a Cozy machine's media sessions (protocol cozy/1).
// WebRTC-direct: the machine is ICE-lite on one passive ICE-TCP candidate and proves itself with
// the DTLS certificate the link pins. No ICE servers, signaling server or proxy: the client
// synthesizes the machine's SDP answer. Dependency-free ES2022, no build step.

export const PROTOCOL = "cozy/1";
const PREFIX = "cozy+webrtc+v1/";
const HEADER = 12; // a binary message is [u32 stream][u64 offset][payload]
const FATAL = new Set(["auth", "expired", "revoked", "scope", "not_found", "runtime_update_required",
  "identity", "link", "webrtc", "mse", "codec", "media", "integrity", "protocol", "closed", "bad_request", "limit"]);

export class CozyError extends Error {
  constructor(code, message, fatal = FATAL.has(code)) { super(message); this.code = code; this.fatal = fatal; }
}

/** Reads a link's fragment: #v=1&a=<ip:port>&f=<64 hex>&c=<capability>&r=<run>&o=<output>[/<index>]. */
export function parseLink(fragment) {
  const q = new URLSearchParams(fragment.replace(/^#/, ""));
  if (q.get("v") !== "1") throw new CozyError("link", q.get("v") ? "This link is newer than this player." : "This link is incomplete.");
  const [address, f, cap, run] = ["a", "f", "c", "r"].map(k => q.get(k));
  if (!/^\[?[^\]]+\]?:\d+$/.test(address ?? "") || !/^[0-9a-f]{64}$/i.test(f ?? "") || !cap || !run) throw new CozyError("link", "This link is incomplete.");
  const [, output, index] = /^(.+?)(?:\/(\d+))?$/.exec(q.get("o") || "video");
  return {address, fingerprint: f.toUpperCase().match(/../g).join(":"), cap, run, output, index: index === undefined ? undefined : +index};
}

/** The machine's SDP answer to an offer: ice-lite, setup:passive, the pinned fingerprint and one passive TCP candidate. */
export function answer(offer, {address, fingerprint}, credential) {
  const mid = /^a=mid:(\S+)/m.exec(offer)?.[1] ?? "0";
  const [, host, port] = /^\[?([^\]]+)\]?:(\d+)$/.exec(address);
  return ["v=0", "o=- 1 1 IN IP4 0.0.0.0", "s=-", "t=0 0", `a=group:BUNDLE ${mid}`, "a=ice-lite",
    "m=application 9 UDP/DTLS/SCTP webrtc-datachannel", "c=IN IP4 0.0.0.0", `a=mid:${mid}`,
    `a=ice-ufrag:${credential}`, `a=ice-pwd:${credential}`, `a=fingerprint:sha-256 ${fingerprint}`,
    "a=setup:passive", "a=sctp-port:5000", "a=max-message-size:65536",
    `a=candidate:1 1 tcp 2130706431 ${host} ${port} typ host tcptype passive`, "a=end-of-candidates", ""].join("\r\n");
}

// The machine keys STUN on the ufrag itself, so ufrag = pwd, in ICE's [A-Za-z0-9+/] alphabet.
function credential() {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  return PREFIX + Array.from(crypto.getRandomValues(new Uint8Array(22)), b => alphabet[b % 62]).join("");
}

/** What this browser lacks to play a MIME type (or anything, without one), or null. */
export function unsupported(type) {
  if (!globalThis.RTCPeerConnection) return new CozyError("webrtc", "This browser has no WebRTC.");
  const Source = globalThis.ManagedMediaSource ?? globalThis.MediaSource;
  if (!Source) return new CozyError("mse", "This browser cannot play streamed video: it has no Media Source Extensions.");
  if (type && !Source.isTypeSupported(type)) return new CozyError("codec", `This browser cannot decode ${type}.`);
  return null;
}

/** Opens a session to the machine at descriptor {address, fingerprint} and presents the capability. */
export async function connect(descriptor, {cap, window = 8 << 20} = {}) {
  const problem = unsupported();
  if (problem) throw problem;
  const pc = new RTCPeerConnection({iceServers: [], bundlePolicy: "max-bundle"});
  const dc = pc.createDataChannel("cozy", {protocol: PROTOCOL});
  dc.binaryType = "arraybuffer";
  let reached = false;
  const opened = new Promise((resolve, reject) => {
    dc.onopen = resolve;
    pc.oniceconnectionstatechange = () => { reached ||= ["connected", "completed"].includes(pc.iceConnectionState); };
    pc.onconnectionstatechange = () => {
      if (["failed", "closed"].includes(pc.connectionState)) reject(reached
        ? new CozyError("identity", "The machine did not prove its identity: its certificate is not the one this link pins.")
        : new CozyError("unreachable", `Cannot reach the machine at ${descriptor.address} over TCP. It may have ended, or this network or browser blocks ICE-TCP.`));
    };
  });
  try {
    await pc.setLocalDescription(await pc.createOffer());
    await pc.setRemoteDescription({type: "answer", sdp: answer(pc.localDescription.sdp, descriptor, credential())});
    await opened;
  } catch (e) {
    pc.close();
    throw e instanceof CozyError ? e : new CozyError("webrtc", `This browser refused the connection: ${e.message}`);
  }
  const session = new Session(pc, dc, window);
  await session.hello(cap);
  return session;
}

/** One authenticated data channel. Requests are multiplexed by id and their bytes by stream. */
export class Session {
  #pc; #dc; #window; #next = 1; #requests = new Map(); #streams = new Map(); #hello = null;
  #released = 0; #granted = 0; #end;
  welcome = null;
  /** Resolves with the CozyError that ended the session. */
  closed = new Promise(resolve => { this.#end = resolve; });

  constructor(pc, dc, window) {
    this.#pc = pc; this.#dc = dc; this.#window = window;
    dc.onmessage = e => typeof e.data === "string" ? this.#control(JSON.parse(e.data)) : this.#data(e.data);
    dc.onclose = () => this.#close(new CozyError("lost", "The connection to the machine closed."));
    pc.addEventListener("connectionstatechange", () => {
      if (["failed", "closed", "disconnected"].includes(pc.connectionState)) this.#close(new CozyError("lost", "The connection to the machine was lost."));
    });
  }

  hello(cap) {
    return new Promise((resolve, reject) => {
      this.#hello = {resolve, reject};
      this.#send({t: "hello", v: 1, cap});
    });
  }

  /** Streams an output's bytes from offset, live, until the run is terminal. The sink may take entry, reset, data(offset, bytes), end, error and close. */
  follow({run, output, index, offset = 0, after = 0}, sink) { return this.#request({t: "follow", run, output, index, offset, after}, sink); }

  /** One byte range of the output's current bytes. With etag, the error `changed` says they moved. */
  get({run, output, index, offset, length, etag}, sink) { return this.#request({t: "get", run, output, index, offset, length, etag}, sink); }

  cancel(id) {
    if (this.#forget(id)) this.#send({t: "cancel", id});
  }

  /** Hands n received bytes back: the machine may send that many more. */
  release(n) {
    this.#released += n;
    if (this.#released + this.#window - this.#granted >= this.#window / 8) this.#grant();
  }

  close(reason = new CozyError("closed", "Closed.")) { this.#close(reason); }

  #grant() {
    this.#granted = this.#released + this.#window;
    this.#send({t: "credit", bytes: this.#granted});
  }

  #send(message) { if (this.#dc?.readyState === "open") this.#dc.send(JSON.stringify(message)); }

  #request(message, sink) {
    const id = this.#next++;
    this.#requests.set(id, sink);
    this.#send({...message, id});
    return id;
  }

  #forget(id) {
    for (const [stream, owner] of this.#streams) if (owner === id) this.#streams.delete(stream);
    return this.#requests.delete(id);
  }

  #close(reason) {
    if (!this.#dc) return;
    const requests = [...this.#requests.values()];
    this.#dc = null;
    this.#requests.clear();
    this.#pc.close();
    this.#hello?.reject(reason);
    for (const sink of requests) sink.close?.(reason);
    this.#end(reason);
  }

  #control(m) {
    if (m.t === "welcome") {
      this.welcome = m;
      this.#hello?.resolve(m);
      this.#hello = null;
      return this.#grant();
    }
    if (m.t === "bye") return this.#close(new CozyError(m.code, BYE[m.code] ?? m.message ?? `The machine closed the session (${m.code}).`, !RETRY.has(m.code)));
    const error = m.t === "error" && new CozyError(m.code, ERROR[m.code] ?? m.message ?? m.code);
    if (error && m.id == null) return this.#close(error);
    const sink = this.#requests.get(m.id);
    if (!sink) return;
    if (m.t === "open") return this.#streams.set(m.stream, m.id);
    if (m.t === "end" || m.t === "error") this.#forget(m.id);
    if (error) sink.error?.(error);
    else sink[m.t]?.(m);
  }

  #data(buffer) {
    const view = new DataView(buffer);
    const payload = new Uint8Array(buffer, HEADER);
    const sink = this.#requests.get(this.#streams.get(view.getUint32(0)));
    if (sink?.data) sink.data(Number(view.getBigUint64(4)), payload);
    else this.release(payload.length);
  }
}

const RETRY = new Set(["evicted", "shutdown"]);
const BYE = {
  auth: "This link is not valid for this machine.",
  expired: "This link has expired. Make a new one with `cozy run play`.",
  revoked: "The key that signed this link was revoked.",
  evicted: "The machine closed this session to make room for another.",
  shutdown: "The machine is shutting down.",
};
const ERROR = {
  scope: "This link does not grant this output.",
  not_found: "The machine has no such run or output.",
  runtime_update_required: "The machine's Runtime predates browser playback. Update it, then reload.",
  unavailable: "The machine cannot serve this output right now.",
};

// ---- MP4: the init segment and its codecs ----

function* boxes(b, start, end) {
  const view = new DataView(b.buffer, b.byteOffset, b.byteLength);
  for (let p = start; p + 8 <= end;) {
    let size = view.getUint32(p), body = p + 8;
    if (size === 1) { if (p + 16 > end) return; size = Number(view.getBigUint64(p + 8)); body += 8; }
    if (size === 0) size = b.length - p;
    if (p + size < body) return;
    yield {type: String.fromCharCode(...b.subarray(p + 4, p + 8)), body, end: p + size};
    p += size;
  }
}

/** The init segment's MIME type with codecs, once ftyp…moov is whole; else null. */
export function initType(b) {
  for (const box of boxes(b, 0, b.length)) {
    if (box.end > b.length) return null;
    if (box.type !== "moov") continue;
    const codecs = [];
    let video = false;
    const child = (x, skip, type) => [...boxes(b, x.body + skip, x.end)].find(c => c.type === type);
    const visit = (x, entries) => {
      for (const y of boxes(b, x.body + (entries ? 8 : 0), x.end)) {
        if (["trak", "mdia", "minf", "stbl"].includes(y.type)) visit(y, false);
        else if (y.type === "stsd") visit(y, true);
        else if (!entries) continue;
        else if (y.type === "mp4a") codecs.push(esds(b, child(y, 28, "esds")));
        else {
          const avcC = /^avc[13]$/.test(y.type) && child(y, 78, "avcC");
          video ||= /^(avc[13]|hvc1|hev1|av01|vp09)$/.test(y.type);
          codecs.push(avcC ? y.type + "." + Array.from(b.subarray(avcC.body + 1, avcC.body + 4), n => n.toString(16).padStart(2, "0")).join("") : y.type);
        }
      }
    };
    visit(box, false);
    return `${video ? "video" : "audio"}/mp4; codecs="${codecs.join(",")}"`;
  }
  return null;
}

// mp4a.<objectTypeIndication>[.<audioObjectType>] from esds' descriptors.
function esds(b, box) {
  if (!box) return "mp4a.40.2";
  let p = box.body + 4;
  const tag = () => { const t = b[p++]; for (let i = 0; i < 4 && b[p++] & 0x80; i++); return t; };
  if (tag() !== 3) return "mp4a.40.2";
  const flags = b[p + 2];
  p += 3 + (flags & 0x80 ? 2 : 0) + (flags & 0x40 ? 1 + b[p + 3] : 0) + (flags & 0x20 ? 2 : 0);
  if (tag() !== 4) return "mp4a.40.2";
  const oti = b[p];
  p += 13;
  if (oti !== 0x40 || tag() !== 5) return "mp4a." + oti.toString(16).toUpperCase();
  const aot = b[p] >> 3;
  return "mp4a.40." + (aot === 31 ? 32 + ((b[p] & 7) << 3 | b[p + 1] >> 5) : aot);
}

const concat = chunks => {
  const out = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
  chunks.reduce((p, c) => (out.set(c, p), p + c.length), 0);
  return out;
};

// ---- Playing an output ----

/** Plays one output of a run in a <video>: live as it grows, seekable once its bytes exist. Reconnects and resumes. */
export function play(video, link, options) { return new Player(video, link, options); }

export class Player extends EventTarget {
  // One entry per revision: its seq, byte span [from, length) and time span [t0, t1). The
  // cursor (seq, pos) is where the follow is; a resumed follow asks for what comes after it.
  entries = [];
  seq = 0;
  pos = 0;
  final = null;
  error = null;
  stats = {bytes: 0, gets: 0, connects: 0, resets: 0, type: "", verified: null}; // bytes: what the follows received
  #video; #link; #options; #media; #session = null; #running = false; #held;
  #follow = 0; #get = 0; #seekTo = null; #waiting = [];

  constructor(video, link, {window = 8 << 20, ahead = 30, verify = false} = {}) {
    super();
    this.#video = video; this.#link = link; this.#options = {window, ahead};
    this.#held = verify ? [] : null; // what the follow delivered, to check against the final sha256
    video.disableRemotePlayback = true; // ManagedMediaSource never opens without it
    video.addEventListener("seeking", () => this.#seek(false));
    video.addEventListener("waiting", () => this.#seek(true));
    video.addEventListener("timeupdate", () => this.#media.pump());
    this.#media = new Media(this);
    this.#run();
  }

  get video() { return this.#video; }
  get ahead() { return this.#options.ahead; }
  get finished() { return this.final !== null && this.pos === this.final.length && !this.#get; }

  close() { this.fail(new CozyError("closed", "Closed.")); }

  fail(error) {
    if (this.error) return;
    this.error = error;
    this.#session?.close();
    this.#status("error", error.message);
  }

  #status(state, detail = "") { this.dispatchEvent(new CustomEvent("status", {detail: {state, detail}})); }

  #target(extra) { return {run: this.#link.run, output: this.#link.output, index: this.#link.index, ...extra}; }

  // A request's error ends the player, or only the session when a new one may do better.
  #failed(error) { if (error.fatal) this.fail(error); else this.#session?.close(error); }

  // Connects while there is something to fetch: the rest of the output, or a seek's range.
  async #run() {
    if (this.#running) return;
    this.#running = true;
    for (let failures = 0; !this.error && (!this.finished || this.#seekTo);) {
      this.#status(failures ? "reconnecting" : "connecting");
      try {
        this.#session = await connect(this.#link, {cap: this.#link.cap, window: this.#options.window});
        this.stats.connects++;
        failures = 0;
        this.#status("connected", this.#session.welcome.machine);
        if (this.#seekTo) this.#fetch(this.#seekTo); else this.#startFollow();
        const reason = await this.#session.closed;
        if (reason.fatal) this.fail(reason);
      } catch (e) {
        if (!(e instanceof CozyError) || e.fatal) this.fail(e instanceof CozyError ? e : new CozyError("internal", String(e)));
        else this.#status("reconnecting", e.message);
      }
      this.#session = null;
      this.#get = 0;
      this.#waiting = [];
      if (this.error || this.finished && !this.#seekTo) break;
      await new Promise(resolve => setTimeout(resolve, Math.min(250 * 2 ** failures++, 8000)));
    }
    this.#running = false;
  }

  #startFollow() {
    const session = this.#session;
    this.#follow = session.follow(this.#target({offset: this.pos, after: this.seq}), {
      entry: m => this.#entry(m),
      reset: m => this.#reset(m, session),
      data: (offset, bytes) => this.#bytes(offset, bytes, session),
      end: m => this.#end(m),
      error: e => this.#failed(e),
    });
  }

  #entry(m) {
    if (this.entries.some(e => e.seq === m.seq)) return;
    // duration_us is the output's whole duration at this revision.
    const t0 = m.appended_from != null ? this.entries.findLast(e => e.length === m.appended_from)?.t1 ?? 0 : 0;
    this.entries.push({seq: m.seq, from: m.appended_from ?? 0, length: m.length, t0, t1: (m.duration_us ?? 0) / 1e6});
    this.dispatchEvent(new CustomEvent("entry", {detail: m}));
    this.#media.pump();
  }

  // The output was replaced: everything held is dropped, and its bytes restart at 0.
  #reset(m, session) {
    this.stats.resets++;
    this.entries = this.entries.filter(e => e.seq >= m.seq);
    this.pos = 0;
    this.#held &&= [];
    session.cancel(this.#get);
    this.#get = 0;
    this.#seekTo = null;
    this.#drop();
    this.#media.restart();
  }

  #drop() {
    for (const [bytes, session] of this.#waiting) session.release(bytes.length);
    this.#waiting = [];
  }

  #bytes(offset, bytes, session) {
    if (offset > this.pos) return this.fail(new CozyError("protocol", `The machine skipped bytes ${this.pos}–${offset}.`));
    this.stats.bytes += bytes.length;
    const skip = Math.min(this.pos - offset, bytes.length);
    if (skip) session.release(skip);
    bytes = bytes.subarray(skip);
    if (!bytes.length) return;
    this.#held?.push(bytes.slice());
    this.pos += bytes.length;
    this.seq = Math.max(this.seq, this.entries.findLast(e => e.length <= this.pos)?.seq ?? 0);
    if (this.#get) this.#waiting.push([bytes, session]); else this.#media.push(bytes, session);
  }

  async #end(m) {
    this.final = {status: m.status, length: m.length ?? this.pos, sha256: m.sha256};
    this.#media.pump();
    if (m.status && m.status !== "completed") this.#status("partial", `The run ${m.status}; this is what it published.`);
    if (this.#held && m.sha256) {
      const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", concat(this.#held)));
      this.stats.verified = Array.from(digest, b => b.toString(16).padStart(2, "0")).join("") === m.sha256;
      if (!this.stats.verified) this.fail(new CozyError("integrity", "The bytes received do not match the run's final sha256."));
    }
    this.#held = null;
  }

  // A seek outside what is buffered, or a stall at a hole in it: fetch the revision that holds
  // the time. A stall at the live edge waits for the follow.
  #seek(stalled) {
    const t = this.#video.currentTime, buffered = this.#video.buffered;
    for (let i = 0; i < buffered.length; i++) if (buffered.start(i) <= t && t < buffered.end(i)) return;
    if (stalled && !(buffered.length && buffered.end(buffered.length - 1) > t + 0.5)) return;
    const e = this.entries.find(x => x.t0 <= t && t < x.t1);
    if (!e || e === this.#seekTo || !this.#media.ready) return;
    // The follow is delivering it, or holds it waiting to be appended.
    if (!this.#get && (e.from <= this.pos && this.pos < e.length || e.length <= this.pos && this.#media.pending)) return;
    this.#seekTo = e;
    if (this.#session) this.#fetch(e); else this.#run();
  }

  // Gets a revision's bytes and follows on from its end at once; the follow's bytes wait for
  // the get's. A replacement meanwhile arrives as the follow's reset, which voids the get.
  #fetch(e) {
    const session = this.#session;
    session.cancel(this.#follow);
    session.cancel(this.#get);
    this.#drop();
    this.#media.discontinuity();
    this.#held = null;
    this.stats.gets++;
    let at = e.from;
    this.#get = session.get(this.#target({offset: e.from, length: e.length - e.from}), {
      data: (offset, bytes) => {
        if (offset !== at) return this.fail(new CozyError("protocol", "The machine sent a range out of order."));
        at += bytes.length;
        this.#media.push(bytes, session);
      },
      end: () => {
        this.#get = 0;
        this.#seekTo = null;
        for (const [bytes] of this.#waiting) this.#media.push(bytes, session);
        this.#waiting = [];
        this.#media.pump();
      },
      error: err => this.#failed(err),
    });
    this.pos = e.length;
    this.seq = e.seq;
    this.#startFollow();
  }
}

// One MediaSource and SourceBuffer. Bytes are appended in arrival order and handed back to their
// session's window once appended, so the machine never runs further ahead than the window.
class Media {
  #player; #source = null; #url = ""; #buffer = null; #opening = false; #queue = []; #appending = [];

  constructor(player) { this.#player = player; this.restart(); }

  get ready() { return this.#buffer !== null; }
  get pending() { return this.#queue.length > 0 || this.#buffer?.updating; }

  restart() {
    this.discontinuity();
    const Source = globalThis.ManagedMediaSource ?? globalThis.MediaSource;
    if (!Source) return;
    URL.revokeObjectURL(this.#url);
    this.#source = new Source();
    this.#buffer = null;
    this.#opening = false;
    this.#source.addEventListener("startstreaming", () => this.pump());
    this.#url = URL.createObjectURL(this.#source);
    this.#player.video.src = this.#url;
  }

  // What is queued is dropped, and the parser restarts at a box boundary.
  discontinuity() {
    for (const {bytes, session} of this.#queue) session.release(bytes.length);
    this.#queue = [];
    if (this.#buffer && this.#source.readyState === "open") this.#buffer.abort();
  }

  push(bytes, session) {
    this.#queue.push({bytes, session});
    if (!this.#buffer && !this.#opening) this.#open();
    this.pump();
  }

  #open() {
    const type = initType(concat(this.#queue.map(q => q.bytes)));
    if (!type) return;
    const problem = unsupported(type);
    if (problem) return this.#player.fail(problem);
    this.#player.stats.type = type;
    this.#opening = true;
    const open = () => {
      this.#buffer = this.#source.addSourceBuffer(type);
      this.#buffer.mode = "segments";
      this.#buffer.addEventListener("updateend", () => {
        for (const {bytes, session} of this.#appending) session.release(bytes.length);
        this.#appending = [];
        this.pump();
      });
      this.pump();
    };
    if (this.#source.readyState === "open") open(); else this.#source.addEventListener("sourceopen", open, {once: true});
  }

  pump() {
    const buffer = this.#buffer, source = this.#source, player = this.#player, video = player.video;
    if (!buffer || buffer.updating || source.readyState === "closed") return;
    // The duration the log announces, so a seek may target what is not yet fetched.
    const known = player.entries.at(-1)?.t1 ?? 0;
    if (source.readyState === "open" && !(source.duration >= known)) source.duration = known;
    if (!this.#queue.length) {
      if (player.finished && source.readyState === "open") source.endOfStream();
      return;
    }
    const end = buffer.buffered.length ? buffer.buffered.end(buffer.buffered.length - 1) : 0;
    if (end - video.currentTime > player.ahead || source.streaming === false) return;
    let count = 0;
    for (let size = 0; count < this.#queue.length && size < 4 << 20;) size += this.#queue[count++].bytes.length;
    try {
      buffer.appendBuffer(count === 1 ? this.#queue[0].bytes : concat(this.#queue.slice(0, count).map(q => q.bytes)));
    } catch (e) {
      if (e.name !== "QuotaExceededError") return player.fail(new CozyError("media", `The browser refused the video: ${e.message}`));
      if (buffer.buffered.length && video.currentTime - 1 > buffer.buffered.start(0)) buffer.remove(0, video.currentTime - 1);
      return;
    }
    this.#appending = this.#queue.splice(0, count);
  }
}
