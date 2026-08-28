#!/usr/bin/env python3
"""No-spend/paid-output verifier for the real Creator H3 workflow receipts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def refuse(message: str) -> None:
    raise SystemExit(f"REFUSED: {message}")


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            refuse(f"JSON key {key!r} appears more than once")
        result[key] = value
    return result


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates,
                          parse_constant=lambda value: refuse(f"JSON contains {value}"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        refuse(f"cannot read {path}: {exc}")


def semantic_digest(value: Any) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        refuse(f"cannot canonicalize review evidence: {exc}")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def document_digest(path: Path) -> str:
    return semantic_digest(load(path))


def write_json(path: Path, value: Any) -> None:
    try:
        data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.staging-", dir=path.parent,
                                         delete=False) as stream:
            staging = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    except (OSError, TypeError, ValueError) as exc:
        try:
            staging.unlink()
        except (NameError, OSError):
            pass
        refuse(f"cannot write {path}: {exc}")


def verify_preflight(probe_path: Path, control_path: Path) -> dict[str, Any]:
    probe, control = load(probe_path), load(control_path)
    if not isinstance(probe, dict) or not isinstance(control, dict):
        refuse("rent probe/show receipts are not JSON objects")
    try:
        rental = os.environ["RENTAL_ID"]
        endpoint = os.environ["H3_ENDPOINT"].strip("/") + "/v1/reference_media_to_video"
        accelerator = os.environ["GPU_SKU"]
        execution = os.environ["EXPECTED_ENDPOINT_EXECUTION"]
        roots = [value.strip() for value in os.environ["EXPECTED_MODEL_ROOTS"].split(",")
                 if value.strip()]
    except KeyError as exc:
        refuse(f"preflight environment omits {exc.args[0]}")
    digest = re.compile(r"sha256:[0-9a-f]{64}").fullmatch
    if not digest(execution) or not roots or len(set(roots)) != len(roots) or \
            not all(digest(value) for value in roots):
        refuse("expected endpoint execution and model roots must be unique lowercase sha256 identities")
    expected = {
        "rental": rental, "state": "ready", "endpoint": endpoint,
        "accelerator": accelerator, "observed_accelerator": accelerator,
        "observed_accelerator_count": 1, "endpoint_execution_digest": execution,
        "model_root_digests": sorted(roots),
    }
    observed = dict(control)
    observed["model_root_digests"] = sorted(control.get("model_root_digests") or [])
    for name, want in expected.items():
        if observed.get(name) != want:
            refuse(f"rent show {name} is {observed.get(name)!r}, expected {want!r}")
    live = ("observed_backend", "observed_worker_instance", "observed_worker_boot_id", "observed_at")
    for name in live:
        if not str(control.get(name, "")).strip():
            refuse(f"rent show carries no {name}")
    for name in ("rental", "state", "endpoint", "accelerator", "observed_accelerator",
                 "observed_accelerator_count", "observed_backend", "observed_worker_instance",
                 "observed_worker_boot_id", "endpoint_execution_digest"):
        if probe.get(name) != control.get(name):
            refuse(f"rent probe {name} is {probe.get(name)!r}, rent show persisted {control.get(name)!r}")
    return control_summary(control)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
    except OSError as exc:
        refuse(f"cannot hash {path}: {exc}")
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
    denominator = float(right or 1)
    if denominator == 0:
        refuse(f"invalid zero-denominator fraction {value!r}")
    return float(left) / denominator


def stream_duration(stream: dict[str, Any], label: str) -> float:
    raw = stream.get("duration")
    try:
        duration = float(raw)
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration <= 0:
        try:
            duration = int(stream["duration_ts"]) * fraction(str(stream["time_base"]))
        except (KeyError, TypeError, ValueError):
            duration = 0.0
    if not math.isfinite(duration) or duration <= 0:
        refuse(f"{label} stream carries no finite positive duration")
    return duration


def stream_start(stream: dict[str, Any], label: str) -> float:
    try:
        start = float(stream.get("start_time") or 0.0)
    except (TypeError, ValueError):
        refuse(f"{label} stream carries an invalid start time")
    if not math.isfinite(start):
        refuse(f"{label} stream carries a non-finite start time")
    return start


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
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    fps = fraction(str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"))
    sample_rate = int(audio.get("sample_rate") or 0)
    if width <= 0 or height <= 0 or frames != expected_frames or \
            abs(fps - 24.0) > 1e-9 or sample_rate != 32000:
        refuse(f"{path} is {frames} frames at {fps} fps/{sample_rate} Hz, expected {expected_frames}/24/32000")
    try:
        duration = float(facts.get("format", {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    video_duration = stream_duration(video, "video")
    audio_duration = stream_duration(audio, "audio")
    video_start = stream_start(video, "video")
    audio_start = stream_start(audio, "audio")
    target = expected_frames / 24.0
    # One video frame or one AAC codec frame is the mechanical endpoint quantum.
    quantum = max(1 / 24, 1024 / 32000) + 1e-6
    for label, observed in (("container", duration), ("video", video_duration),
                            ("audio", audio_duration)):
        if not math.isfinite(observed) or abs(observed - target) > quantum:
            refuse(f"{path} {label} duration {observed:.6f}s differs from {target:.6f}s beyond one codec quantum")
    if abs(video_start) > quantum or abs(audio_start) > quantum:
        refuse(f"{path} streams start at video={video_start:.6f}s/audio={audio_start:.6f}s")
    video_endpoint, audio_endpoint = video_start + video_duration, audio_start + audio_duration
    av_delta = abs(video_endpoint - audio_endpoint)
    if av_delta > quantum:
        refuse(f"{path} A/V stream endpoints differ by {av_delta:.6f}s, beyond one codec quantum")
    command("ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-")
    integrity = verify_integrity(path, duration)
    return {"digest": sha256(path), "width": width, "height": height,
            "frames": frames, "fps": fps,
            "container_duration": duration, "video_duration": video_duration,
            "audio_duration": audio_duration, "video_start": video_start,
            "audio_start": audio_start, "video_endpoint": video_endpoint,
            "audio_endpoint": audio_endpoint, "av_endpoint_delta": av_delta,
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


def confined(path: Path, root: Path) -> Path:
    if path.is_absolute():
        refuse(f"acceptance index path {path} is absolute")
    candidate = root / path
    try:
        candidate.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        refuse(f"saved output {path} escapes acceptance root {root}")
    return candidate


def saved_path(receipt: dict[str, Any], output: str, root: Path) -> Path:
    for row in receipt.get("saved_outputs", []):
        if row.get("output") == output:
            path = confined(Path(row["path"]), root)
            if sha256(path) != row.get("digest"):
                refuse(f"downloaded {output} no longer matches its receipt")
            return path
    refuse(f"step {receipt.get('ordinal')} has no downloaded {output}")
    raise AssertionError


def indexed_receipts(bundle: Path, manifest: dict[str, Any],
                     state: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[int, str]]:
    rows = manifest.get("steps")
    if not isinstance(rows, list) or len(rows) != len(state.get("steps", [])):
        refuse("download manifest step index does not match workflow state")
    receipts: list[dict[str, Any]] = []
    materialized: dict[int, str] = {}
    seen: set[int] = set()
    for row in rows:
        expected_keys = {"ordinal", "receipt_path", "receipt_digest",
                         "materialized_path", "materialized_digest"}
        if not isinstance(row, dict) or "ordinal" not in row or not set(row) <= expected_keys:
            refuse("download manifest step row is not the closed compact index shape")
        try:
            ordinal = int(row["ordinal"])
        except (KeyError, TypeError, ValueError):
            refuse("download manifest has a step with no integer ordinal")
        if ordinal < 1 or ordinal > len(rows) or ordinal in seen:
            refuse(f"download manifest step ordinal {ordinal} is absent or duplicated")
        seen.add(ordinal)
        step = state["steps"][ordinal - 1]
        materialized_path = row.get("materialized_path")
        if bool(materialized_path) != bool(row.get("materialized_digest")):
            refuse(f"step {ordinal} materialized path/digest pair is incomplete")
        if materialized_path:
            path = confined(Path(materialized_path), bundle)
            digest = sha256(path)
            if digest != row.get("materialized_digest") or \
                    digest != step.get("materialized_submission_digest"):
                refuse(f"step {ordinal} materialized bytes do not match workflow state")
            materialized[ordinal] = digest
        elif step.get("materialized"):
            refuse(f"step {ordinal} is materialized but its exact bytes were not exported")
        receipt_path = row.get("receipt_path")
        if bool(receipt_path) != bool(row.get("receipt_digest")):
            refuse(f"step {ordinal} receipt path/digest pair is incomplete")
        if receipt_path:
            path = confined(Path(receipt_path), bundle)
            if sha256(path) != row.get("receipt_digest"):
                refuse(f"step {ordinal} receipt bytes do not match the manifest")
            receipt = load(path)
            if int(receipt.get("ordinal", 0)) != ordinal:
                refuse(f"step {ordinal} receipt declares another ordinal")
            receipts.append(receipt)
        elif step.get("child_request_id"):
            refuse(f"step {ordinal} has a child but no receipt export")
    if seen != set(range(1, len(rows) + 1)):
        refuse("download manifest step ordinals are not the complete ordered set")
    return sorted(receipts, key=lambda row: row["ordinal"]), materialized


def indexed_file(bundle: Path, value: Any, expected_path: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "digest"} or \
            value.get("path") != expected_path:
        refuse(f"acceptance bundle index for {expected_path} is not one exact path/digest ref")
    path = confined(Path(expected_path), bundle)
    if sha256(path) != value.get("digest"):
        refuse(f"acceptance bundle file {expected_path} differs from its index digest")
    return path


def indexed_controls(bundle: Path, path: Path) -> dict[str, dict[str, Any]]:
    document = load(path)
    if not isinstance(document, dict):
        refuse("rental-control.json is not a rental-id map")
    expected_keys = {
        "endpoint", "accelerator", "observed_accelerator", "observed_accelerator_count",
        "observed_backend", "observed_worker_instance", "observed_worker_boot_id", "observed_at",
        "snapshot_path", "control_snapshot_digest", "control_snapshot_length",
        "endpoint_execution_digest", "artifact_object_set_digest", "model_root_digests",
        "endpoint_release_id", "descriptor_digest", "environment_spec_digest",
        "installed_environment_receipt_digest", "placement_set_digest", "binding_plan_digests",
    }
    controls: dict[str, dict[str, Any]] = {}
    for rental_id, value in document.items():
        if not isinstance(value, dict) or set(value) != expected_keys or not value.get("snapshot_path"):
            refuse(f"rental control {rental_id!r} is not the closed compact summary")
        snapshot = confined(Path(value["snapshot_path"]), bundle)
        digest = sha256(snapshot)
        if digest != value.get("control_snapshot_digest"):
            refuse(f"rental control {rental_id!r} snapshot digest changed")
        try:
            length = snapshot.stat().st_size
        except OSError as exc:
            refuse(f"cannot stat rental control snapshot {snapshot}: {exc}")
        if length != int(value.get("control_snapshot_length", -1)):
            refuse(f"rental control {rental_id!r} snapshot length changed")
        summary = control_summary(value)
        summary["rental"] = rental_id
        controls[rental_id] = summary
    return controls


def control_summary(value: dict[str, Any]) -> dict[str, Any]:
    names = (
        "rental", "state", "endpoint", "accelerator", "observed_accelerator",
        "observed_accelerator_count", "observed_backend", "observed_worker_instance",
        "observed_worker_boot_id", "observed_at",
        "control_snapshot_digest", "control_snapshot_length", "endpoint_execution_digest",
        "endpoint_release_id",
        "artifact_object_set_digest", "model_root_digests", "descriptor_digest",
        "environment_spec_digest", "installed_environment_receipt_digest",
        "placement_set_digest", "binding_plan_digests",
    )
    return {name: value.get(name) for name in names if value.get(name) not in (None, "")}


def restart_evidence(root: Path, workflow_id: str) -> dict[str, Any]:
    before_submit = load(root / "submit-before-kill.json")
    after_submit = load(root / "submit-after-restart.json")
    before = load(root / "service-before-kill.json")
    killed = load(root / "service-after-kill.json")
    restarted = load(root / "service-after-restart.json")
    if before_submit.get("workflow") != workflow_id or after_submit.get("workflow") != workflow_id:
        refuse("restart submission did not replay the exact workflow id")
    if before.get("service") != "up" or killed.get("service") != "down" or \
            restarted.get("service") != "up":
        refuse("restart receipts do not show LocalService up, killed, then up")
    try:
        before_pid, restarted_pid = int(before["pid"]), int(restarted["pid"])
    except (KeyError, TypeError, ValueError):
        refuse("restart receipts carry no concrete service pids")
    if before_pid <= 0 or restarted_pid <= 0:
        refuse("restart receipts carry non-positive service pids")
    return {
        "workflow_id": workflow_id,
        "before_pid": before_pid,
        "restarted_pid": restarted_pid,
        "before_submit_digest": document_digest(root / "submit-before-kill.json"),
        "after_submit_digest": document_digest(root / "submit-after-restart.json"),
        "before_service_digest": document_digest(root / "service-before-kill.json"),
        "killed_service_digest": document_digest(root / "service-after-kill.json"),
        "restarted_service_digest": document_digest(root / "service-after-restart.json"),
    }


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


def transition_psnr(image: Path, video: Path) -> float | str:
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
    value = float(psnr(reference, candidate))
    if math.isnan(value):
        refuse("transition PSNR is NaN")
    return "infinite" if math.isinf(value) else value


def verify_image(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image
        with Image.open(path) as image:
            if image.format != "PNG" or image.width <= 0 or image.height <= 0:
                refuse(f"{path} is not one non-empty PNG image")
            width, height = image.width, image.height
            image.verify()
    except (ImportError, OSError, SyntaxError) as exc:
        refuse(f"cannot verify continuation image {path}: {exc}")
    return {"digest": sha256(path), "width": width, "height": height}


def verify_single(root: Path) -> dict[str, Any]:
    control = control_summary(load(root / "control" / "rental-control.json"))
    videos = list((root / "media").glob("*.mp4"))
    images = list((root / "media").glob("*.png"))
    if len(videos) != 1 or len(images) != 1:
        refuse("single proof needs one MP4 and one continuation PNG")
    video, continuation = verify_video(videos[0], 345), verify_image(images[0])
    if (video["width"], video["height"]) != (continuation["width"], continuation["height"]):
        refuse("single-shot video and continuation image have different geometry")
    report = {"mode": "single", "control": control,
              "rental_probe_digest": document_digest(root / "control" / "rental-probe.json"),
              "videos": [video],
              "continuation_frame": {**continuation,
                                     "path": images[0].relative_to(root).as_posix()}}
    write_json(root / "verification.json", report)
    return report


def verify_bundle(mode: str, root: Path) -> dict[str, Any]:
    bundle = root / "download"
    manifest = load(bundle / "download-manifest.json")
    expected_manifest_keys = {"version", "workflow", "workflow_plan", "creative_plan",
                              "steps", "rental_control"}
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_keys or \
            manifest.get("version") != 1:
        refuse("download manifest is not the closed compact version-1 index")
    workflow_path = indexed_file(bundle, manifest["workflow"], "workflow.json")
    workflow_plan_path = indexed_file(bundle, manifest["workflow_plan"], "workflow-plan.json")
    creative_plan_path = indexed_file(bundle, manifest["creative_plan"], "creative-plan.json")
    rental_control_path = indexed_file(bundle, manifest["rental_control"], "rental-control.json")
    state = load(workflow_path)
    receipts, materialized = indexed_receipts(bundle, manifest, state)
    composition = load(root / "composition.json")
    compose_record = load(root / "compose.json")
    creative_digest = str(composition.get("creative_plan_digest", ""))
    if not creative_digest or state.get("creative_plan_digest") != creative_digest or \
            compose_record.get("creative_plan_digest") != creative_digest or \
            sha256(creative_plan_path) != creative_digest or \
            load(creative_plan_path) != composition.get("creative_plan") or \
            load(workflow_plan_path) != composition.get("workflow_plan"):
        refuse("composition, submit record, and workflow disagree on creative-plan identity")
    controls = indexed_controls(bundle, rental_control_path)
    if len(controls) != 1:
        refuse("workflow bundle does not bind exactly one H3 rental control")
    preflight_control = control_summary(load(root / "control" / "rental-control.json"))
    downloaded_control = next(iter(controls.values()))
    for name in ("endpoint", "accelerator", "observed_accelerator",
                 "observed_accelerator_count", "observed_backend",
                 "observed_worker_instance", "observed_worker_boot_id",
                 "endpoint_execution_digest", "model_root_digests"):
        if downloaded_control.get(name) != preflight_control.get(name):
            refuse(f"downloaded rental control {name} differs from the preflight readback")
    probe_digest = document_digest(root / "control" / "rental-probe.json")
    expected_shots = {"two": 2, "eight": 8}.get(mode, 0)
    if mode == "cancel":
        if state.get("status") != "canceled":
            refuse("cancellation bundle is not canceled")
        if [row["ordinal"] for row in receipts] != [1]:
            refuse("cancellation after shot one published an unexpected later request receipt")
        first = receipts[0]
        event_types = [event.get("type") for event in first.get("events", [])]
        if first.get("lifecycle", {}).get("status") != "canceled" or \
                "request.accepted" not in event_types or "request.canceled" not in event_types:
            refuse("cancellation receipt does not prove one accepted shot settled canceled")
        later = state["steps"][1:]
        if len(later) != 2 or any(step.get("child_request_id") or step.get("outputs")
                                  for step in later):
            refuse("shot two or assembly published after workflow cancellation")
        report = {"mode": mode, "status": state["status"], "later_child_absent": True,
                  "absent_steps": [2, 3], "creative_plan_digest": creative_digest,
                  "canceled_request": first.get("lifecycle", {}).get("request_id"),
                  "composition_digest": document_digest(root / "composition.json"),
                  "workflow_plan_digest": sha256(workflow_plan_path),
                  "preflight_control": preflight_control, "rental_probe_digest": probe_digest,
                  "materialized_submission_digests": materialized,
                  "rental_control": controls}
        write_json(root / "verification.json", report)
        return report
    if state.get("status") != "succeeded" or len(receipts) != expected_shots + 1:
        refuse(f"{mode} bundle is not one succeeded {expected_shots + 1}-step workflow")
    shot_receipts = receipts[:expected_shots]
    assembly = receipts[-1]
    videos = [verify_video(saved_path(row, "video", bundle), 345) for row in shot_receipts]
    continuation_paths = [saved_path(row, "continuation_frame", bundle) for row in shot_receipts]
    continuations = [verify_image(path) for path in continuation_paths]
    for index, (video, image) in enumerate(zip(videos, continuations, strict=True), start=1):
        if (video["width"], video["height"]) != (image["width"], image["height"]):
            refuse(f"shot {index} video and continuation image have different geometry")
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
        score = transition_psnr(continuation_paths[index - 1],
                                saved_path(shot_receipts[index], "video", bundle))
        transitions.append({"from_step": index, "to_step": index + 1,
                            "continuation_digest": prior_digest, "first_frame_psnr_db": score})
    final_frames = 689 if expected_shots == 2 else 2753
    final_video = verify_video(saved_path(assembly, "video", bundle), final_frames)
    result = assembly["lifecycle"].get("result") or {}
    segment_digests = [row.get("digest") for row in result.get("segments", [])]
    shot_digests = [output_digest(row, "video") for row in shot_receipts]
    if segment_digests != shot_digests:
        refuse("assembly segment order/digests differ from ordered shot outputs")
    if int(result.get("output_frames", 0)) != final_frames or \
            int(result.get("replay_frames_removed", -1)) != expected_shots - 1:
        refuse("assembly receipt frame arithmetic is wrong")
    quantum = int(result.get("audio_codec_frame_samples", 0))
    delta = abs(int(result.get("av_endpoint_delta_samples", quantum + 1)))
    if quantum <= 0 or delta > quantum:
        refuse("assembly A/V endpoint exceeds its recorded codec-frame quantum")
    report = {"mode": mode, "status": state["status"], "shots": videos,
              "continuation_frames": continuations,
              "transitions": transitions, "final": final_video,
              "segment_order": segment_digests, "creative_plan_digest": creative_digest,
              "composition_digest": document_digest(root / "composition.json"),
              "workflow_plan_digest": sha256(workflow_plan_path),
              "preflight_control": preflight_control, "rental_probe_digest": probe_digest,
              "materialized_submission_digests": materialized,
              "rental_control": controls}
    if mode == "two":
        report["restart_recovery"] = restart_evidence(root, str(state.get("workflow_id", "")))
    write_json(root / "verification.json", report)
    return report


def control_identity(root: Path) -> tuple[str, str, str, str, tuple[str, ...]]:
    verification = load(root / "verification.json")
    control = verification.get("control")
    if control is not None:
        row = control
    else:
        controls = verification.get("rental_control", {})
        if len(controls) != 1:
            refuse(f"{root} does not bind exactly one H3 rental control")
        row = next(iter(controls.values()))
    return (str(row.get("rental", "")), str(row.get("endpoint", "")),
            str(row.get("observed_accelerator", "")),
            str(row["endpoint_execution_digest"]), tuple(sorted(row["model_root_digests"])))


def expected_control_identity() -> tuple[str, str, str, str, tuple[str, ...]]:
    try:
        return (os.environ["RENTAL_ID"],
                os.environ["H3_ENDPOINT"].strip("/") + "/v1/reference_media_to_video",
                os.environ["GPU_SKU"], os.environ["EXPECTED_ENDPOINT_EXECUTION"],
                tuple(sorted(value.strip() for value in os.environ["EXPECTED_MODEL_ROOTS"].split(",")
                             if value.strip())))
    except KeyError as exc:
        refuse(f"progressive gate environment omits {exc.args[0]}")


def gate_stage(mode: str, root: Path) -> None:
    path = root / "verification.json"
    if not path.is_file() or load(path).get("mode") != mode:
        refuse(f"{root} has no successful {mode} verification receipt")
    if control_identity(root) != expected_control_identity():
        refuse(f"{mode} prerequisite used another rental, endpoint, GPU, execution, or root set")


MANUAL_FORMAT = "cozy.h3.ManualReview/1"


def manual_header(manual: dict[str, Any]) -> str:
    if not isinstance(manual, dict) or set(manual) != {"format", "reviewer", "single", "two", "eight"} or \
            manual.get("format") != MANUAL_FORMAT:
        refuse(f"manual review format is not {MANUAL_FORMAT}")
    schemas = {
        "single": {"verification_digest", "viewed", "listened", "notes"},
        "two": {"verification_digest", "viewed", "listened", "transition_accepted", "notes"},
        "eight": {"verification_digest", "viewed", "listened", "transitions_accepted", "notes"},
    }
    for name, keys in schemas.items():
        if not isinstance(manual.get(name), dict) or set(manual[name]) != keys or \
                not isinstance(manual[name].get("notes"), str):
            refuse(f"manual {name} review is not the closed {MANUAL_FORMAT} section")
    reviewer = str(manual.get("reviewer", "")).strip()
    if not reviewer:
        refuse("manual review has no reviewer")
    return reviewer


def require_review(manual: dict[str, Any], name: str, verification: Path,
                   flags: tuple[str, ...]) -> dict[str, Any]:
    return require_review_digest(manual, name, document_digest(verification), flags)


def require_review_digest(manual: dict[str, Any], name: str, expected: str,
                          flags: tuple[str, ...]) -> dict[str, Any]:
    section = manual.get(name)
    if not isinstance(section, dict):
        refuse(f"manual review has no {name} section")
    if section.get("verification_digest") != expected:
        refuse(f"manual {name} review binds {section.get('verification_digest')!r}, expected {expected}")
    for flag in flags:
        if section.get(flag) is not True:
            refuse(f"manual {name} review has not accepted {flag}")
    return section


def gate_eight(single: Path, two: Path, cancel: Path, manual_path: Path) -> dict[str, Any]:
    for root in (single, two, cancel):
        if not (root / "verification.json").is_file():
            refuse(f"{root} has no successful automated verification receipt")
    modes = [load(root / "verification.json").get("mode") for root in (single, two, cancel)]
    if modes != ["single", "two", "cancel"]:
        refuse(f"prerequisite receipt modes are {modes!r}, expected single/two/cancel")
    identities = {control_identity(root) for root in (single, two, cancel)}
    if len(identities) != 1:
        refuse("single/two/cancel proofs used different rental, endpoint, GPU, execution, or roots")
    if next(iter(identities)) != expected_control_identity():
        refuse("eight-shot environment differs from the prerequisite rental/execution evidence")
    cancel_verification = load(cancel / "verification.json")
    if cancel_verification.get("later_child_absent") is not True or \
            cancel_verification.get("absent_steps") != [2, 3]:
        refuse("cancellation proof did not establish later-child absence")
    manual = load(manual_path)
    reviewer = manual_header(manual)
    require_review(manual, "single", single / "verification.json", ("viewed", "listened"))
    require_review(manual, "two", two / "verification.json",
                   ("viewed", "listened", "transition_accepted"))
    identity = next(iter(identities))
    return {
        "format": "cozy.h3.EightShotPrerequisite/1",
        "reviewer": reviewer,
        "rental": identity[0],
        "endpoint": identity[1],
        "observed_accelerator": identity[2],
        "endpoint_execution_digest": identity[3],
        "model_root_digests": list(identity[4]),
        "verification_digests": {
            "single": document_digest(single / "verification.json"),
            "two": document_digest(two / "verification.json"),
            "cancel": document_digest(cancel / "verification.json"),
        },
        "manual_review_digest": semantic_digest(manual),
    }


def finalize_eight(root: Path, manual_path: Path) -> dict[str, Any]:
    gate_path = root / "prerequisite-gate.json"
    gate = load(gate_path)
    if gate.get("format") != "cozy.h3.EightShotPrerequisite/1":
        refuse("eight-shot run carries no valid prerequisite gate")
    # Re-run every automated video/receipt check immediately before binding the
    # human verdict, so a modified download cannot inherit an older green report.
    verify_bundle("eight", root)
    identity = control_identity(root)
    if (gate.get("rental"), gate.get("endpoint"), gate.get("observed_accelerator"),
            gate.get("endpoint_execution_digest"), tuple(gate.get("model_root_digests") or [])) != identity:
        refuse("eight-shot run used a different rental, endpoint, GPU, execution, or model-root set")
    manual = load(manual_path)
    reviewer = manual_header(manual)
    if reviewer != gate.get("reviewer"):
        refuse("final manual reviewer differs from the prerequisite reviewer")
    verification_digests = gate.get("verification_digests")
    if not isinstance(verification_digests, dict):
        refuse("eight-shot prerequisite gate carries no verification digest set")
    require_review_digest(manual, "single", str(verification_digests.get("single", "")),
                          ("viewed", "listened"))
    require_review_digest(manual, "two", str(verification_digests.get("two", "")),
                          ("viewed", "listened", "transition_accepted"))
    section = require_review(manual, "eight", root / "verification.json",
                             ("viewed", "listened"))
    transitions = section.get("transitions_accepted")
    if transitions != [True] * 7:
        refuse("manual eight review must accept each of the seven transitions separately")
    acceptance = {
        "format": "cozy.h3.ManualAcceptance/1",
        "reviewer": reviewer,
        "prerequisite_gate_digest": document_digest(gate_path),
        "verification_digest": document_digest(root / "verification.json"),
        "manual_review_digest": semantic_digest(manual),
        "review": section,
    }
    write_json(root / "manual-acceptance.json", acceptance)
    return acceptance


def main(argv: list[str]) -> None:
    if len(argv) == 4 and argv[1] == "preflight":
        verify_preflight(Path(argv[2]), Path(argv[3]))
        print("preflight: exact endpoint, live GPU readback, execution, and model-root set match")
        return
    if len(argv) == 3 and argv[1] == "digest":
        print(document_digest(Path(argv[2])))
        return
    if len(argv) == 4 and argv[1] == "gate-stage" and argv[2] in {"single", "two"}:
        gate_stage(argv[2], Path(argv[3]))
        print(f"{argv[2]} prerequisite gate passed")
        return
    if len(argv) == 3 and argv[1] == "single":
        verify_single(Path(argv[2]))
        print("single-shot verification passed; manual watch/listen still required")
        return
    if len(argv) == 4 and argv[1] == "bundle" and argv[2] in {"two", "cancel", "eight"}:
        verify_bundle(argv[2], Path(argv[3]))
        print(f"{argv[2]} workflow verification passed; manual watch/listen still required where applicable")
        return
    if len(argv) == 6 and argv[1] == "gate-eight":
        print(json.dumps(gate_eight(*(Path(value) for value in argv[2:])),
                         ensure_ascii=False, sort_keys=True))
        return
    if len(argv) == 4 and argv[1] == "finalize-eight":
        acceptance = finalize_eight(Path(argv[2]), Path(argv[3]))
        print(json.dumps(acceptance, ensure_ascii=False, sort_keys=True))
        return
    refuse("usage: verify.py preflight PROBE CONTROL | digest JSON | "
           "gate-stage single|two DIR | single DIR | "
           "bundle two|cancel|eight DIR | "
           "gate-eight SINGLE TWO CANCEL MANUAL | finalize-eight EIGHT MANUAL")


if __name__ == "__main__":
    main(sys.argv)
