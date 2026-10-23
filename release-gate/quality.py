"""Output quality of one gate, one of analyze.py's CHECKS, with cozy-eval's two cheapest instruments on one CLIP
(ViT-B/16, OpenAI weights, CPU):
  - adherence: cosine similarity between each render (or 3 frames of a video) and its prompt (prompts.tsv, which
    gate.sh's `cell` writes). Every render must clear FLOOR: noise, a flat field, a broken decode or a text encoder
    that ignores the prompt lands far under any real render;
  - sharpness: CLIP-IQA's "A sharp image." against "A blurry image.". Single renders vary too much to judge alone (a
    transparent object on grey reads soft), so the gate's median must hold against the last passed gates' medians.
Never a same-seed pixel diff: a kernel swap moves the pixels (26 dB) without changing the quality.
Measured on the 10-23 gate outputs: adherence 0.325-0.430 for renders against their own prompts, 0.06-0.31 against
another gate's prompts, 0.16-0.18 for noise and grey. Sharpness medians 0.80 for the renders, 0.07 blurred (6 px),
0.04 blocky (1/16), 0.16 noise.
    quality.py <gate dir> [<baseline file>]     the same verdict alone"""
import glob, json, os, subprocess, sys, tempfile
import numpy as np
from PIL import Image

MODEL = 'openai/clip-vit-base-patch16'
FLOOR = 0.25
# The sharpness median may fall this far under the lowest of the last KEEP passed medians.
BUDGET, KEEP = 0.25, 5
SKIP = ('birefnet', 'describer', 'seedvr2')  # cutouts on white, text, upscales of H3's own frames
_clip = None


def clip():
    global _clip
    if _clip is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        torch.set_num_threads(2)
        _clip = (CLIPModel.from_pretrained(MODEL).eval(), CLIPProcessor.from_pretrained(MODEL), torch)
    return _clip


def judge(ims, prompt):
    """(adherence, sharpness) per image."""
    model, proc, torch = clip()
    with torch.no_grad():
        o = model(**proc(text=[prompt, 'A sharp image.', 'A blurry image.'], images=ims, return_tensors='pt', padding=True, truncation=True))
    adherence = (o.image_embeds @ o.text_embeds[:1].T).squeeze(1)
    sharp = o.logits_per_image[:, 1:].softmax(-1)[:, 0]
    return [(round(float(a), 4), round(float(s), 4)) for a, s in zip(adherence, sharp)]


def rgb(path):
    im = Image.open(path)
    if im.mode in ('RGBA', 'LA'):  # transparency: judge the render on mid-grey, not on its alpha
        bg = Image.new('RGBA', im.size, (128, 128, 128, 255)); bg.alpha_composite(im.convert('RGBA')); im = bg
    return im.convert('RGB')


def frames(path, n=3):
    d = tempfile.mkdtemp()
    dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path],
                               capture_output=True, text=True).stdout.strip() or 0)
    for i in range(n):
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{dur * (i + 1) / (n + 1):.3f}', '-i', path, '-frames:v', '1', f'{d}/{i}.png'])
    return [Image.open(f).convert('RGB') for f in sorted(glob.glob(f'{d}/*.png'))]


def check(gate):
    out, base = gate['out'], gate.get('baseline') or {}
    p = os.path.join(out, 'prompts.tsv')
    prompts = dict(l.rstrip('\n').split('\t', 1) for l in open(p) if '\t' in l) if os.path.exists(p) else {}
    fails, notes, cells = [], [], {}
    for line in open(os.path.join(out, 'cells.tsv')):
        name, _, number = line.rstrip('\n').split('\t')[:3]
        if name not in prompts or not number.isdigit() or name.split('-')[0] in SKIP:
            continue
        files = sorted(glob.glob(os.path.expanduser(f'~/.cozy/outputs/*/{number}-*')))
        ims = [rgb(f) for f in files if f.endswith(('.png', '.webp', '.jpg'))] + [fr for f in files if f.endswith('.mp4') for fr in frames(f)]
        if ims:
            cells[name] = judge(ims, prompts[name])
            worst = min(a for a, _ in cells[name])
            if worst < FLOOR:
                fails.append(f'quality {name}: adherence {worst:.3f} under {FLOOR}: not what its prompt asked for (noise, flat, broken)')
    if not cells:
        return fails, notes
    adh = float(np.median([a for v in cells.values() for a, _ in v]))
    med = round(float(np.median([s for v in cells.values() for _, s in v])), 4)
    hist = base.setdefault('quality', [])
    line = f'quality: {len(cells)} cells, adherence median {adh:.3f}, sharpness median {med:.3f}'
    if hist:
        limit = min(h['sharpness'] for h in hist[-KEEP:]) - BUDGET
        line += f' (last passed {", ".join(format(h["sharpness"], ".3f") for h in hist[-KEEP:])}; floor {limit:.3f})'
        if med < limit:
            fails.append(line + ': the renders went soft'); line = None
    if line:
        notes.append(line)
    hist.append({'sharpness': med, 'adherence': round(adh, 4), 'cells': cells, 'label': open(os.path.join(out, 'label')).read().strip()})
    del hist[:-KEEP]
    return fails, notes


if __name__ == '__main__':
    out = sys.argv[1].rstrip('/')
    base = json.load(open(sys.argv[2])) if len(sys.argv) > 2 and os.path.exists(sys.argv[2]) else {}
    fails, notes = check({'out': out, 'baseline': base})
    for n in notes: print('note', n)
    for f in fails: print('FAIL', f)
    print('QUALITY', 'FAIL' if fails else 'PASS')
    sys.exit(1 if fails else 0)
