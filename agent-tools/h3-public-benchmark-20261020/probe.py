"""Actual child-only capacity/identity check, before the H3 payload transfer."""
import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import time

from contract import check_capacity, check_spec, deadline_budget, uuid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--harness", required=True)
    p.add_argument("--wheel", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--deadline", type=int, required=True)
    args = p.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    check_spec(spec)
    deadline_budget(time.time(), args.deadline)
    h3 = runpy.run_path(args.harness, run_name="frozen_h3_capacity")
    cohort = h3["installed_provenance"](args.wheel, spec["native"]["sha256"], spec["native"]["source"])
    torch, _, versions = h3["load_cohort"](spec["dependencies"]["versions"]["torch"])
    count = torch.cuda.device_count()
    h3["device_namespace"](count, "cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.init()
    torch.cuda.synchronize(0)
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    memory = h3["memory_state"]()
    work = Path(args.work).resolve(strict=True)
    targets = [work / "source-store", work / "fixtures"]
    same_filesystem = all(not target.exists() and target.parent.stat().st_dev == work.stat().st_dev for target in targets)
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader,nounits"],
                                  text=True, timeout=15)
    rows = [line.split(",", 3) for line in raw.splitlines() if line.strip()]
    matching = [row for row in rows if len(row) == 4 and uuid(row[1].strip()) == uuid(props.uuid)]
    if len(matching) != 1:
        raise ValueError("selected Torch UUID must match one physical NVML row")
    row = [value.strip() for value in matching[0]]
    result = {"schema": "h3-capacity/1", "visible_count": count, "device": "cuda:0", "gpu_uuid": str(props.uuid),
              "name": props.name, "compute_capability": [props.major, props.minor],
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "physical": {"gpu": int(row[0]), "uuid": row[1], "name": row[2], "driver": row[3]},
              "gpu_available": free, "gpu_total": total, "host_available": memory["available"],
              "host": memory, "disk_available": shutil.disk_usage(work).free,
              "filesystem_device": os.stat(work).st_dev, "same_work_filesystem": same_filesystem,
              "versions": versions, "cohort": cohort, "checked_at": time.time()}
    # Both future source-store and fixture directories are children of this fresh plain work directory.
    check_capacity(result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
