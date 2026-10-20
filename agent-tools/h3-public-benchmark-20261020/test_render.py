"""Synthetic zip/PEP723 tests: no wheel install, import, native execution, or real payload."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
import zipfile

import render

HERE = Path(__file__).resolve().parent


def make_wheel(path, name, source, pure=False):
    prefix = name.replace("-", "_")
    info = f"{prefix}-9.9.9.dist-info/"
    image = bytearray(24)
    image[:6] = b"\x7fELF\x02\x01"
    image[18:20] = b"\x3e\x00"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(info + "METADATA", f"Name: {name}\nVersion: 9.9.9\nRequires-Python: >=3.12\n")
        z.writestr(info + "WHEEL", f"Root-Is-Purelib: {'true' if pure else 'false'}\nTag: cp312-abi3-manylinux_2_28_x86_64\n")
        z.writestr(prefix + "/__init__.py", "# synthetic, intentionally not executable as a native package\n")
        if name == "cozy-runtime":
            z.writestr(prefix + "/_build_provenance.py", f'COMMIT = "{source}"\n')
            for module in ("_gemm", "_norm", "_fa3"): z.writestr(prefix + f"/_kernels/{module}.abi3.so", image)
        else:
            z.writestr(prefix + "/_ext.abi3.so", image)
    return {"source": source, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class Render(unittest.TestCase):
    def test_pure_runtime_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.whl"
            expected = make_wheel(path, "cozy-runtime", "a" * 40, pure=True)
            with self.assertRaisesRegex(ValueError, "native wheel required"):
                render.wheel(path, "cozy-runtime", expected)

    def test_synthetic_render_keeps_paths_index_signature_and_full_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime, native = root / "runtime.whl", root / "tensorfs.whl"
            rt = make_wheel(runtime, "cozy-runtime", "a" * 40)
            nt = make_wheel(native, "tensorfs", "b" * 40)
            review = root / "review.json"; review.write_text('{"synthetic":true}')
            candidate = root / "candidate.json"
            candidate.write_text(json.dumps({"runtime": rt, "native": nt,
                "pair_receipt_sha256": hashlib.sha256(review.read_bytes()).hexdigest()}))
            requirements = root / "requirements.txt"
            requirements.write_text("torch==2.14.0+cu130\nsafetensors==0.8.0\nfastsafetensors==0.5.0\nnumpy==2.5.4\n")
            repo = Path(subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"], text=True).strip())
            output = root / "payload/h3.py"
            args = SimpleNamespace(candidate=candidate, pair_receipt=review, runtime_wheel=runtime,
                native_wheel=native, requirements=requirements, creator_repo=repo, output=output)
            render.generate(args)
            text = output.read_text()
            header = text.split("# ///", 2)[1].splitlines()[1:]
            metadata = tomllib.loads("\n".join(line.removeprefix("# ") for line in header))
            self.assertEqual(metadata["tool"]["uv"]["sources"]["cozy-runtime"], {"path": "../runtime.whl"})
            self.assertEqual(metadata["tool"]["uv"]["sources"]["tensorfs"], {"path": "../tensorfs.whl"})
            self.assertEqual(metadata["tool"]["uv"]["sources"]["torch"], {"index": "pytorch-cu130"})
            self.assertTrue(metadata["tool"]["uv"]["index"][0]["explicit"])
            module = ast.parse(text)
            main = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "main")
            self.assertEqual([a.arg for a in main.args.kwonlyargs],
                             ["stage_client", "metadata_bundle", "native_wheel", "deadline_unix", "out"])
            held = json.loads(output.with_suffix(".manifest.json").read_text())
            self.assertEqual(len(held["spec"]["schedule"]), 60)
            self.assertEqual(held["payload_sha256"], hashlib.sha256(text.encode()).hexdigest())
            self.assertNotIn("torch", __import__("sys").modules)
            with self.assertRaisesRegex(ValueError, "fresh payload paths"): render.generate(args)


if __name__ == "__main__": unittest.main()
