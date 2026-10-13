#!/usr/bin/env python3
"""Extract the selected Runtime wheel's agent; inspect version without booting it."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def extract(wheel: Path, target: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist() if name.endswith('.data/scripts/cozy-machine')]
        records = [name for name in archive.namelist() if name.endswith('.dist-info/RECORD')]
        if len(members) != 1 or len(records) != 1:
            raise ValueError('Runtime wheel must contain exactly one bundled agent and RECORD')
        member = members[0]
        data = archive.read(member)
        rows = [row for row in csv.reader(io.StringIO(archive.read(records[0]).decode())) if row[0] == member]
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
        if len(rows) != 1 or rows[0][1:] != ['sha256=' + digest, str(len(data))]:
            raise ValueError('bundled agent does not match wheel RECORD')
        if data[:6] != b'\x7fELF\x02\x01' or int.from_bytes(data[18:20], 'little') != 62:
            raise ValueError('bundled agent must be a Linux x86-64 ELF executable')
    target.write_bytes(data)
    target.chmod(0o755)
    version = json.loads(subprocess.check_output([str(target.resolve()), 'version'], text=True))
    required = {'hub-access/1', 'runtime-update/1', 'machine-bootstrap/1'}
    if version.get('name') != 'cozy-machine' or not required.issubset(version.get('capabilities', [])):
        raise ValueError('bundled agent lacks the current machine operation contracts')
    print(json.dumps({'wheel': str(wheel), 'agent_sha256': hashlib.sha256(data).hexdigest(), 'version': version}))


if __name__ == '__main__':
    extract(Path(sys.argv[1]), Path(sys.argv[2]))
