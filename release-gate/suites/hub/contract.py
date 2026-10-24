"""The Hub's API to an anonymous caller against its contract (tensorhub api/openapi.yaml): a path open to anyone
answers 200 with a body its response schema admits; a bearer-only path refuses with 401. This is the one place in
release-gate that speaks HTTP to the Hub by hand: it is the contract test, for the paths no cozy verb reaches as a
person uses it (cells.sh drives the rest through `cozy`). Prints `check<TAB>PASS|FAIL<TAB>what` rows for expect.
    contract.py <hub> <openapi.yaml> paths
    contract.py <hub> <openapi.yaml> yanked <org> <distribution> <version>   the simple index marks it data-yanked"""
import json
import re
import sys
import urllib.error
import urllib.request

import jsonschema
import yaml

HUB, DOC, MODE = sys.argv[1].rstrip('/'), sys.argv[2], sys.argv[3]
spec = yaml.safe_load(open(DOC))
# The first-party root namespace and a package, model and SKU every Hub release serves.
PATHS = [('/v1/software', {}), ('/.well-known/oauth-protected-resource', {}),
         ('/v1/auth/.well-known/openid-configuration', {}), ('/v1/packages', {}), ('/v1/packages/{org}/{name}', {'org': '_', 'name': 'sdxl'}),
         ('/v1/models', {}), ('/v1/models/{org}/{name}', {'org': '_', 'name': 'sdxl'}), ('/v1/rental-skus', {}),
         ('/v1/rental-skus/{name}', {'name': 'cpu'}), ('/v1/execution-environment', {}), ('/v1/index/{org}/simple/', {'org': '_'}),
         ('/v1/index/{org}/simple/{distribution}/', {'org': '_', 'distribution': 'sdxl'})]


def row(check, ok, what):
    print(f'{check}\t{"PASS" if ok else "FAIL"}\t{what}')


def resolve(node, seen=()):
    """The schema with every $ref inlined and OpenAPI 3.0's `nullable` as JSON Schema's null type."""
    if isinstance(node, list):
        return [resolve(n, seen) for n in node]
    if not isinstance(node, dict):
        return node
    if '$ref' in node:
        ref = node['$ref']
        if ref in seen:
            return {}
        target = spec
        for part in ref.lstrip('#/').split('/'):
            target = target[part]
        return resolve(target, seen + (ref,))
    out = {k: resolve(v, seen) for k, v in node.items() if k not in ('nullable', 'example', 'discriminator', 'xml')}
    if node.get('nullable') and 'type' in out:
        out['type'] = [out['type'], 'null']
    return out


def get(path):
    req = urllib.request.Request(HUB + path, headers={'Accept': 'application/json, text/html'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.headers.get('content-type', ''), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get('content-type', ''), e.read()


if MODE == 'paths':
    for template, args in PATHS:
        path = template.format(**args)
        status, ctype, body = get(path)
        op = (spec['paths'].get(template) or {}).get('get')
        if op is None:
            row(f'hub {path}', False, f'{template} is not in the contract document')
            continue
        security = op.get('security')
        if security and not any(not s for s in security):  # bearer only: an anonymous caller is refused
            row(f'hub {path}', status == 401, f'{status} to an anonymous caller (its contract: bearer only)')
            continue
        if status != 200:
            row(f'hub {path}', False, f'{status}: {body[:200]!r}')
            continue
        content = ((op.get('responses') or {}).get('200') or {}).get('content') or {}
        schema = (content.get('application/json') or {}).get('schema')
        if schema is None or 'json' not in ctype:
            row(f'hub {path}', True, f'200 {ctype} ({len(body)} bytes)')
            continue
        try:
            jsonschema.validate(json.loads(body), resolve(schema))
            row(f'hub {path}', True, f'200, the body matches its schema ({len(body)} bytes)')
        except (ValueError, jsonschema.ValidationError) as e:
            row(f'hub {path}', False, f'200, but the body breaks its schema: {str(e).splitlines()[0][:300]}')
elif MODE == 'yanked':
    org, dist, version = sys.argv[4:7]
    status, _, body = get(f'/v1/index/{org}/simple/{dist}/')
    links = re.findall(r'<a [^>]*>[^<]*</a>', body.decode('utf-8', 'replace'))
    mine = [a for a in links if f'-{version}' in a or f'_{version}' in a]
    ok = status == 200 and mine and all('data-yanked' in a for a in mine)
    row('hub index data-yanked', bool(ok), f'{status}: {len(mine)} file(s) of {dist} {version}, '
        f'{sum("data-yanked" in a for a in mine)} marked data-yanked')
