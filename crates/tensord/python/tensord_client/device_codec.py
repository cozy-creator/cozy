"""The selected SDK's encoder for one executor's registered frames: image, audio and video.

Embedded by Rust and started once per executor: one JSON request per stdin line, one JSON
answer per stdout line; it ends on stdin EOF. It imports the selected installed SDK; no
model or codec implementation is copied.
"""
import json
import os
import stat
import sys
import uuid

from cozy_runtime.author._codec import FRAME_MEDIA_TYPES, encode_frame


def regular(directory, name):
    """The spool file `name`'s status: a regular file, never a link or a path elsewhere."""
    if not name or name != os.path.basename(name) or name in (".", ".."):
        raise ValueError("SDK output has no single owned spool name")
    handle = os.open(name, os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory)
    try:
        status = os.fstat(handle)
        if not stat.S_ISREG(status.st_mode):
            raise ValueError("SDK output is not a regular file")
        return handle, status
    except BaseException:
        os.close(handle)
        raise


def read_regular(directory, name):
    handle, _ = regular(directory, name)
    try:
        with open(f"/proc/self/fd/{handle}", "rb") as stream:
            return stream.read()
    finally:
        os.close(handle)


def encode(request, directory):
    """Each frame encoded into the spool name the machine gave it: its media type, by handle."""
    media = {}
    for frame in request["frames"]:
        raw = read_regular(directory, frame["raw"])
        if len(raw) != frame["raw_bytes"]:
            raise ValueError("registered raw frame length changed")
        name = frame["name"]
        if not name or name != os.path.basename(name) or name in (".", ".."):
            raise ValueError("SDK frame has no owned output binding")
        encoded = encode_frame(frame["codec"], frame["facts"], raw)
        temporary = ".codec-" + uuid.uuid4().hex
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600,
                     dir_fd=directory)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
        os.rename(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        media[frame["handle"]] = FRAME_MEDIA_TYPES[frame["codec"]]
    return {"media": media}


def main():
    for line in sys.stdin:
        try:
            request = json.loads(line)
            directory = os.open(request["spool"], os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                answer = encode(request, directory)
            finally:
                os.close(directory)
        except Exception as exc:  # one refused output never ends the executor's encoder
            answer = {"error": f"{type(exc).__name__}: {exc}"[:2000]}
        sys.stdout.write(json.dumps(answer, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
