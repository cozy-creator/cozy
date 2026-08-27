#!/usr/bin/env python3
"""No-spend/paid-output verifier for the real Creator H3 workflow receipts."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


def refuse(message: str) -> None:
    raise SystemExit(f"REFUSED: {message}")


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        refuse(f"cannot read {path}: {exc}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def command(*argv: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(argv, check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        refuse(f"{' '.join(argv[:2])} failed: {detail.strip()}")


def command_bytes(*argv: str) -> bytes:
    try:
        return subprocess.run(argv, check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", "replace")
        refuse(f"{' '.join(argv[:2])} failed: {str(detail or exc).strip()}")
        raise AssertionError


def probe(path: Path) -> dict[str, Any]:
    return json.loads(command(
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ).stdout)


def fraction(value: str) -> float:
    left, _, right = value.partition("/")
    return float(left) / float(right or 1)


def verify_video(path: Path, expected_frames: int) -> dict[str, Any]:
    if not path.is_file():
        refuse(f"video is absent: {path}")
    facts = probe(path)
    video = next((row for row in facts.get("streams", []) if row.get("codec_type") == "video"), None)
    audio = next((row for row in facts.get("streams", []) if row.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        refuse(f"{path} does not carry both video and audio")
    if video.get("codec_name") != "h264" or audio.get("codec_name") != "aac":
        refuse(f"{path} codecs are {video.get('codec_name')}/{audio.get('codec_name')}, want h264/aac")
    frames = int(video.get("nb_read_frames") or video.get("nb_frames") or 0)
    fps = fraction(str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"))
    sample_rate = int(audio.get("sample_rate") or 0)
    if frames != expected_frames or abs(fps - 24.0) > 1e-9 or sample_rate != 32000:
        refuse(f"{path} is {frames} frames at {fps} fps/{sample_rate} Hz, expected {expected_frames}/24/32000")
    duration = float(facts.get("format", {}).get("duration") or 0.0)
    target = expected_frames / 24.0
    # One video frame or one AAC codec frame is the mechanical endpoint quantum.
    if abs(duration - target) > max(1 / 24, 1024 / 32000):
        refuse(f"{path} duration {duration:.6f}s differs from {target:.6f}s beyond one codec quantum")
    command("ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-")
    integrity = verify_integrity(path, duration)
    return {"path": str(path), "frames": frames, "fps": fps, "duration": duration,
            "sample_rate": sample_rate, "integrity": integrity}


def verify_integrity(path: Path, duration: float) -> dict[str, Any]:
    try:
        import numpy as np
        from cozy_eval.integrity import PASS, output_integrity
        from cozy_eval.metrics.signal import score
    except ImportError as exc:
        refuse(f"install cozy-eval plus numpy for integrity verification: {exc}")
    windows = []
    for start in (0.0, max(0.0, duration / 2 - 0.35), max(0.0, duration - 0.7)):
        raw = command_bytes(
            "ffmpeg", "-v", "error", "-ss", f"{start:.6f}", "-i", str(path),
            "-vf", "scale=512:288", "-frames:v", "16", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-",
        )
        stride = 512 * 288 * 3
        usable = len(raw) // stride * stride
        if usable:
            windows.append(np.frombuffer(raw[:usable], np.uint8).reshape(-1, 288, 512, 3))
    if not windows:
        refuse(f"{path} decoded no integrity frames")
    sampled = np.concatenate(windows)
    checked = output_integrity(sampled, pairs=min(8, max(1, len(sampled) // 2)))
    if checked.verdict != PASS:
        refuse(f"{path} output integrity is {checked.summary()}")
    unique = len({hashlib.sha256(frame.tobytes()).digest() for frame in sampled})
    if unique <= 1:
        refuse(f"{path} sampled frames are frozen")
    signal = score(path).metrics()
    if not all(math.isfinite(float(value)) for value in signal.values()):
        refuse(f"{path} signal metrics contain NaN or infinity")
    return {"output_integrity": checked.summary(), "sampled_frames": len(sampled),
            "unique_sampled_frames": unique, "signal": signal}


def saved_path(receipt: dict[str, Any], output: str) -> Path:
    for row in receipt.get("saved_outputs", []):
        if row.get("output") == output:
            path = Path(row["path"])
            if sha256(path) != row.get("digest"):
                refuse(f"downloaded {output} no longer matches its receipt")
            return path
    refuse(f"step {receipt.get('ordinal')} has no downloaded {output}")
    raise AssertionError


def output_digest(receipt: dict[str, Any], output: str) -> str:
    for row in receipt["lifecycle"].get("outputs", []):
        if row.get("output_id") == output:
            return str(row["digest"])
    refuse(f"step {receipt.get('ordinal')} has no {output} output fact")
    raise AssertionError


def first_frame(path: Path) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        refuse(f"numpy is required for transition verification: {exc}")
    facts = probe(path)
    stream = next(row for row in facts["streams"] if row.get("codec_type") == "video")
    width, height = int(stream["width"]), int(stream["height"])
    raw = command_bytes("ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-")
    if len(raw) != width * height * 3:
        refuse(f"{path} first-frame decode is incomplete")
    return np.frombuffer(raw, np.uint8).reshape(height, width, 3)


def transition_psnr(image: Path, video: Path) -> float:
    try:
        import numpy as np
        from PIL import Image
        from cozy_eval.metrics.reference import psnr
    except ImportError as exc:
        refuse(f"Pillow and cozy-eval are required for transition verification: {exc}")
    reference = np.asarray(Image.open(image).convert("RGB"))
    candidate = first_frame(video)
    if reference.shape != candidate.shape:
        refuse(f"transition shapes differ: {reference.shape} vs {candidate.shape}")
    return float(psnr(reference, candidate))


def verify_single(root: Path) -> dict[str, Any]:
    control = load(root / "control" / "rental-control.json")
    videos = list((root / "media").glob("*.mp4"))
    images = list((root / "media").glob("*.png"))
    if len(videos) != 1 or len(images) != 1:
        refuse("single proof needs one MP4 and one continuation PNG")
    report = {"mode": "single", "control": control, "videos": [verify_video(videos[0], 345)],
              "continuation_frame": str(images[0])}
    (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify_bundle(mode: str, root: Path) -> dict[str, Any]:
    manifest = load(root / "download-manifest.json")
    state = manifest["workflow"]
    receipts = sorted(manifest["requests"], key=lambda row: row["ordinal"])
    expected_shots = {"two": 2, "eight": 8}.get(mode, 0)
    if mode == "cancel":
        if state.get("status") != "canceled":
            refuse("cancellation bundle is not canceled")
        assembly = state["steps"][-1]
        if assembly.get("child_request_id") or assembly.get("outputs"):
            refuse("a child or output published after workflow cancellation")
        report = {"mode": mode, "status": state["status"], "later_child_absent": True,
                  "rental_control": manifest.get("rental_control", {})}
        (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    if state.get("status") != "succeeded" or len(receipts) != expected_shots + 1:
        refuse(f"{mode} bundle is not one succeeded {expected_shots + 1}-step workflow")
    shot_receipts = receipts[:expected_shots]
    assembly = receipts[-1]
    videos = [verify_video(saved_path(row, "video"), 345) for row in shot_receipts]
    transitions = []
    for index in range(1, expected_shots):
        prior_digest = output_digest(shot_receipts[index - 1], "continuation_frame")
        step = state["steps"][index]
        bound = next((row for row in step.get("resolved_bindings", [])
                      if row.get("field_path") == "first_frame"), None)
        asset = next((row for row in step.get("materialized_assets", [])
                      if row.get("field_path") == "first_frame"), None)
        if bound is None or asset is None or bound.get("digest") != prior_digest or asset.get("digest") != prior_digest:
            refuse(f"step {index + 1} does not bind the exact prior continuation_frame")
        score = transition_psnr(saved_path(shot_receipts[index - 1], "continuation_frame"),
                                saved_path(shot_receipts[index], "video"))
        transitions.append({"from_step": index, "to_step": index + 1,
                            "continuation_digest": prior_digest, "first_frame_psnr_db": score})
    final_frames = 689 if expected_shots == 2 else 2753
    final_video = verify_video(saved_path(assembly, "video"), final_frames)
    result = assembly["lifecycle"].get("result") or {}
    segment_digests = [row.get("digest") for row in result.get("segments", [])]
    shot_digests = [output_digest(row, "video") for row in shot_receipts]
    if segment_digests != shot_digests:
        refuse("assembly segment order/digests differ from ordered shot outputs")
    if int(result.get("output_frames", 0)) != final_frames or int(result.get("replay_frames_removed", -1)) != expected_shots - 1:
        refuse("assembly receipt frame arithmetic is wrong")
    quantum = int(result.get("audio_codec_frame_samples", 0))
    delta = abs(int(result.get("av_endpoint_delta_samples", quantum + 1)))
    if quantum <= 0 or delta > quantum:
        refuse("assembly A/V endpoint exceeds its recorded codec-frame quantum")
    report = {"mode": mode, "status": state["status"], "shots": videos,
              "transitions": transitions, "final": final_video,
              "segment_order": segment_digests, "rental_control": manifest.get("rental_control", {})}
    (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def control_identity(root: Path) -> tuple[str, tuple[str, ...]]:
    verification = load(root / "verification.json")
    control = verification.get("control")
    if control is not None:
        return str(control["endpoint_execution_digest"]), tuple(sorted(control["model_root_digests"]))
    controls = verification.get("rental_control", {})
    if len(controls) != 1:
        refuse(f"{root} does not bind exactly one H3 rental control")
    row = next(iter(controls.values()))["control"]
    return str(row["endpoint_execution_digest"]), tuple(sorted(row["model_root_digests"]))


def gate_eight(single: Path, two: Path, cancel: Path, manual_path: Path) -> None:
    for root in (single, two, cancel):
        if not (root / "verification.json").is_file():
            refuse(f"{root} has no successful automated verification receipt")
    identities = {control_identity(root) for root in (single, two, cancel)}
    if len(identities) != 1:
        refuse("single/two/cancel proofs used different endpoint execution or model roots")
    if load(cancel / "verification.json").get("later_child_absent") is not True:
        refuse("cancellation proof did not establish later-child absence")
    manual = load(manual_path)
    required = ("single_viewed", "single_listened", "two_shot_viewed",
                "two_shot_listened", "two_shot_transition_accepted")
    if not str(manual.get("reviewer", "")).strip() or not all(manual.get(name) is True for name in required):
        refuse("manual single/two watch-listen and transition acceptance are incomplete")
    print("eight-shot gate: prior automated and manual receipts use one exact execution")


def main(argv: list[str]) -> None:
    if len(argv) == 3 and argv[1] == "single":
        verify_single(Path(argv[2]))
        print("single-shot verification passed; manual watch/listen still required")
        return
    if len(argv) == 4 and argv[1] == "bundle" and argv[2] in {"two", "cancel", "eight"}:
        verify_bundle(argv[2], Path(argv[3]))
        print(f"{argv[2]} workflow verification passed; manual watch/listen still required where applicable")
        return
    if len(argv) == 6 and argv[1] == "gate-eight":
        gate_eight(*(Path(value) for value in argv[2:]))
        return
    refuse("usage: verify.py single DIR | bundle two|cancel|eight DIR | gate-eight SINGLE TWO CANCEL MANUAL")


if __name__ == "__main__":
    main(sys.argv)
