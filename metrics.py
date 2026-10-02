"""Automatic metrics from Table 2 of "Rethinking Global Text Conditioning in Diffusion Transformers":
CLIP score, ImageReward, PickScore, HPSv3 -- computed for every (prompt idx, condition) image of a run.

Stages (run with the /venv/metrics environment, after generation is finished):
    python metrics.py light     # CLIP (L/14, H/14, bigG/14), PickScore, ImageReward -- one pass, all models on GPU at once
    python metrics.py hpsv3     # HPSv3 (Qwen2-VL-7B reward model), resumable
    python metrics.py analyze   # per-condition means, paired diffs vs condition 0 with bootstrap CIs

Score conventions (chosen to match the scale of Table 2, FLUX schnell: PS 22.9, CLIP 35.6, IR 10.2, HPSv3 11.3):
    clip_*      = 100 * cos(image, text)                         (torchmetrics-style CLIP score; backbone not stated in the paper)
    pickscore   = logit_scale.exp() * cos  (= 100 * cos)          (official PickScore)
    imagereward = raw ImageReward-v1.0 score; the paper reports it x10 -> see analyze
    hpsv3       = mu of the HPSv3 output (official inference: rewards[:, 0])
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor

REPO_DIR = Path(__file__).resolve().parent
N_COND = 5
COND_NAMES = {0: 'x', 1: 'x*', 2: 'x* @ norm(x)', 3: 'x @ norm(x*)', 4: 'x, gate*1.03'}


def load_items(img_dir, limit=None):
    df = pd.read_csv(REPO_DIR / 'dataset/coco5000.csv', sep='|', names=['idx', 'text'])[:limit]
    items = [(int(idx), c, text, img_dir / f'{idx}_{c}.png') for idx, text in df.values.tolist() for c in range(N_COND)]
    missing = [str(p) for *_, p in items if not p.exists()]
    if missing:
        sys.exit(f'{len(missing)} images missing, e.g. {missing[:3]}')
    return items


# ---------------------------------------------------------------- light: CLIP x3, PickScore, ImageReward
# All five models use the same input: bicubic resize to 224, center crop, OpenAI-CLIP mean/std.
CLIP_TRANSFORM = Compose([
    Resize(224, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


class ImageDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        idx, c, text, path = self.items[i]
        with Image.open(path) as im:
            return CLIP_TRANSFORM(im.convert('RGB')), idx, c


@torch.no_grad()
def run_light(args):
    from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPTokenizer
    import open_clip
    import ImageReward as RM

    dev = args.device
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    items = load_items(args.img_dir, args.limit)
    prompts = {idx: text for idx, _, text, _ in items}
    idx_order = sorted(prompts)
    texts = [prompts[i] for i in idx_order]
    row_of = {idx: r for r, idx in enumerate(idx_order)}

    # each scorer: (encode_images(x) -> normalized feats, normalized text feats [n_prompts, d], multiplier)
    scorers = {}
    t0 = time.time()

    m = CLIPModel.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16).to(dev).eval()
    tok = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    tf = torch.cat([m.get_text_features(**tok(texts[i:i + 512], padding=True, truncation=True, max_length=77,
                                                return_tensors='pt').to(dev)) for i in range(0, len(texts), 512)])
    scorers['clip_L14'] = (m.get_image_features, torch.nn.functional.normalize(tf.float(), dim=-1), 100.0)

    for name, hub in [('clip_H14', 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'),
                      ('clip_bigG14', 'hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')]:
        m = open_clip.create_model(hub, precision='fp16', device=dev).eval()
        tok = open_clip.get_tokenizer(hub)
        tf = torch.cat([m.encode_text(tok(texts[i:i + 512]).to(dev)) for i in range(0, len(texts), 512)])
        scorers[name] = (m.encode_image, torch.nn.functional.normalize(tf.float(), dim=-1), 100.0)

    m = AutoModel.from_pretrained('yuvalkirstain/PickScore_v1', torch_dtype=torch.float16).to(dev).eval()
    proc = AutoProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    tf = torch.cat([m.get_text_features(**proc(text=texts[i:i + 512], padding=True, truncation=True, max_length=77,
                                                 return_tensors='pt').to(dev)) for i in range(0, len(texts), 512)])
    scorers['pickscore'] = (m.get_image_features, torch.nn.functional.normalize(tf.float(), dim=-1),
                            m.logit_scale.exp().item())

    ir = RM.load('ImageReward-v1.0', device=dev, download_root=os.path.expanduser('~/.cache/ImageReward')).eval()
    # tokenize every prompt once, exactly as ImageReward.score does (max_length=35)
    ir_tok = ir.blip.tokenizer(texts, padding='max_length', truncation=True, max_length=35, return_tensors='pt').to(dev)
    print(f'models loaded in {time.time() - t0:.0f}s; GPU mem {torch.cuda.memory_allocated() / 1e9:.1f} GB', flush=True)

    dl = DataLoader(ImageDataset(items), batch_size=args.batch_size, num_workers=args.workers,
                    pin_memory=True, persistent_workers=False, prefetch_factor=4)
    out = {k: [] for k in ['idx', 'cond', *scorers, 'imagereward']}
    t0 = time.time()
    for b, (x, idx, c) in enumerate(dl):
        x = x.to(dev, non_blocking=True)
        rows = torch.tensor([row_of[int(i)] for i in idx], device=dev)
        out['idx'] += idx.tolist()
        out['cond'] += c.tolist()
        xh = x.half()
        for name, (enc, tfeat, mult) in scorers.items():
            f = torch.nn.functional.normalize(enc(xh).float(), dim=-1)
            out[name] += (mult * (f * tfeat[rows]).sum(-1)).tolist()
        # ImageReward: same computation as ImageReward.score, batched, fp32 (+TF32)
        emb = ir.blip.visual_encoder(x)
        att = torch.ones(emb.shape[:-1], dtype=torch.long, device=dev)
        h = ir.blip.text_encoder(ir_tok.input_ids[rows], attention_mask=ir_tok.attention_mask[rows],
                                 encoder_hidden_states=emb, encoder_attention_mask=att, return_dict=True)
        r = (ir.mlp(h.last_hidden_state[:, 0, :].float()) - ir.mean) / ir.std
        out['imagereward'] += r.squeeze(-1).tolist()
        if b % 10 == 0:
            done = len(out['idx'])
            print(f'light: {done}/{len(items)}  {done / (time.time() - t0):.1f} img/s', flush=True)

    df = pd.DataFrame(out)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / 'light.csv', index=False)
    print(f'saved {len(df)} rows -> {args.out_dir / "light.csv"} in {time.time() - t0:.0f}s')


# ---------------------------------------------------------------- HPSv3
class HPSDataset(Dataset):
    """Builds the exact HPSv3 chat input (as in HPSv3RewardInferencer.prepare_batch) in DataLoader workers."""

    def __init__(self, items, processor):
        self.items, self.processor = items, processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def collate(self, batch):
        from hpsv3.dataset.data_collator_qwen import INSTRUCTION, prompt_with_special_token
        from hpsv3.dataset.utils import process_vision_info
        px = 256 * 28 * 28
        messages = [[{'role': 'user', 'content': [
            {'type': 'image', 'image': str(path), 'min_pixels': px, 'max_pixels': px},
            {'type': 'text', 'text': INSTRUCTION.format(text_prompt=text) + prompt_with_special_token},
        ]}] for _, _, text, path in batch]
        image_inputs, _ = process_vision_info(messages)
        enc = self.processor(text=self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                             images=image_inputs, padding=True, return_tensors='pt', videos_kwargs={'do_rescale': True})
        return dict(enc), [b[0] for b in batch], [b[1] for b in batch]


@torch.no_grad()
def run_hpsv3(args):
    from hpsv3 import HPSv3RewardInferencer

    items = load_items(args.img_dir, args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / 'hpsv3.csv'
    done = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = set(zip(prev.idx, prev.cond))
    todo = [it for it in items if (it[0], it[1]) not in done]
    print(f'hpsv3: {len(done)} done, {len(todo)} to go', flush=True)
    if not todo:
        return

    inf = HPSv3RewardInferencer(device=args.device)
    assert inf.use_special_tokens
    # rm_head is fp32 and HPSv3 relies on a CUDA-only autocast to feed it bf16 hidden states; casting the input
    # explicitly gives the same result on GPU and makes CPU runs (smoke tests) work too
    inf.model.rm_head.register_forward_pre_hook(lambda mod, inp: tuple(t.float() for t in inp))
    ds = HPSDataset(todo, inf.processor)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, collate_fn=ds.collate, prefetch_factor=4)
    t0, n, buf = time.time(), 0, []
    for b, (enc, idx, c) in enumerate(dl):
        enc = {k: v.to(args.device, non_blocking=True) for k, v in enc.items()}
        mu = inf.model(return_dict=True, **enc)['logits'][:, 0].float().tolist()
        buf += list(zip(idx, c, mu))
        n += len(idx)
        if b % 10 == 0 or n == len(todo):
            pd.DataFrame(buf, columns=['idx', 'cond', 'hpsv3']).to_csv(
                out_path, mode='a', header=not out_path.exists(), index=False)
            buf = []
            print(f'hpsv3: {n}/{len(todo)}  {n / (time.time() - t0):.2f} img/s', flush=True)
    if buf:
        pd.DataFrame(buf, columns=['idx', 'cond', 'hpsv3']).to_csv(out_path, mode='a', header=not out_path.exists(), index=False)


# ---------------------------------------------------------------- analysis
def run_analyze(args, n_boot=2000, seed=0):
    df = pd.read_csv(args.out_dir / 'light.csv')
    hp = args.out_dir / 'hpsv3.csv'
    if hp.exists():
        df = df.merge(pd.read_csv(hp).drop_duplicates(['idx', 'cond'], keep='last'), on=['idx', 'cond'], how='left')
    df['imagereward_x10'] = df['imagereward'] * 10
    metrics = [m for m in ['pickscore', 'clip_L14', 'clip_H14', 'clip_bigG14', 'imagereward_x10', 'hpsv3'] if m in df]
    rng = np.random.default_rng(seed)
    lines = [f'# Metrics, {args.img_dir.name} ({df.idx.nunique()} prompts)\n',
             'Paper, Table 2, FLUX schnell: PickScore 22.9 -> 23.1, CLIP 35.6 -> 35.8, IR(x10) 10.2 -> 11.0, HPSv3 11.3 -> 11.8 (Aesthetics)\n',
             '## Means\n', '| cond | ' + ' | '.join(metrics) + ' |', '|---' * (len(metrics) + 1) + '|']
    means = df.groupby('cond')[metrics].mean()
    for c, r in means.iterrows():
        lines.append(f'| {c}: {COND_NAMES[c]} | ' + ' | '.join(f'{r[m]:.3f}' for m in metrics) + ' |')
    lines += ['\n## Paired difference vs cond 0: mean [95% bootstrap CI over prompts], win rate %\n',
              '| cond | ' + ' | '.join(metrics) + ' |', '|---' * (len(metrics) + 1) + '|']
    wide = {m: df.pivot(index='idx', columns='cond', values=m) for m in metrics}
    for c in range(1, N_COND):
        cells = []
        for m in metrics:
            d = (wide[m][c] - wide[m][0]).to_numpy()
            d = d[~np.isnan(d)]
            bm = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
            lo, hi = np.percentile(bm, [2.5, 97.5])
            cells.append(f'{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], {100 * (d > 0).mean():.0f}%')
        lines.append(f'| {c}: {COND_NAMES[c]} | ' + ' | '.join(cells) + ' |')
    text = '\n'.join(lines) + '\n'
    (args.out_dir / 'summary.md').write_text(text)
    print(text)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['light', 'hpsv3', 'analyze'])
    ap.add_argument('--img-dir', type=Path, default=REPO_DIR / 'generations/coco5000')
    ap.add_argument('--out-dir', type=Path, default=REPO_DIR / 'results/coco5000')
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=None, help='only the first N prompts (smoke tests)')
    args = ap.parse_args()
    if args.batch_size is None:
        args.batch_size = {'light': 256, 'hpsv3': 32, 'analyze': 0}[args.stage]
    {'light': run_light, 'hpsv3': run_hpsv3, 'analyze': run_analyze}[args.stage](args)
