import tensorfs,struct,json,sys
root='/var/lib/tensorfs'
tensorfs.gc(root)
store=tensorfs.Store.open(root)
manifest=sys.argv[1]
with store.acquire_cozytensors(manifest) as lease:
    parts=[r for r in store.walk_cozytensors(manifest) if r['kind']=='part']
    assert len(parts)==1
    part=parts[0]
    buf=bytearray(part['length'])
    lease.read_into(part['id'],part['length'],0,part['length'],buf)
    values=struct.unpack('<512f',buf)
    assert all(v==float(sys.argv[2]) for v in values),values[:3]
print(json.dumps({'manifest':manifest,'values':len(values),'first':values[0],'last':values[-1]}))
