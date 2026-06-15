"""Space alignment evaluation for the siamese video/audio model.

For every sample we pool one embedding per modality at several depths:

  - "mapper": mean of the mapper output (CLIP->d_model / Whisper->d_model)
  - "enc_i":  mean of the M2M encoder hidden state after layer i

To compare the two modalities at encoder depth we run each branch through the
encoder on its own (no shared language prompt) and pool only over the modality
positions, dropping the trailing EOS — otherwise the shared prompt/EOS tokens
would inflate the similarity.

At each depth we report cross-modal retrieval, relational similarity (RSA +
CKA), and -- for the mapper -- hubness and canonical correlations.

    python eval_space_alignment.py --checkpoint_path CKPT --metadata_file test.tsv
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import skew, spearmanr

log = logging.getLogger("eval_space_alignment")

WHISPER_FRAMES = 3000


def load_model(checkpoint, device):
    from multimodalhugs.models.siamese_multimodal_embedder.modeling_siamese_multimodal_embedder import (
        SiameseMultiModalEmbedderModel,
    )
    model = SiameseMultiModalEmbedderModel.from_pretrained(checkpoint)
    return model.eval().to(device)


def resolve_preprocessors(checkpoint, video_override, audio_override):
    video, audio = video_override, audio_override
    config = Path(checkpoint) / "config.json"
    if config.exists() and (video is None or audio is None):
        cfg = json.loads(config.read_text())
        video = video or cfg.get("pretrained_feature_extractor", "openai/clip-vit-base-patch32")
        audio = audio or cfg.get("audio_pretrained_feature_extractor", "openai/whisper-medium")
    return video or "openai/clip-vit-base-patch32", audio or "openai/whisper-medium"


def build_processors(video_pp, audio_pp, skip_frames_stride):
    from multimodalhugs.processors.video_modality_processor import VideoModalityProcessor
    from multimodalhugs.processors.speech_modality_processor import SpeechModalityProcessor
    video = VideoModalityProcessor(
        custom_preprocessor_path=video_pp,
        skip_frames_stride=skip_frames_stride,
        join_chw=False,
    )
    audio = SpeechModalityProcessor(custom_preprocessor_path=audio_pp)
    return video, audio


@torch.no_grad()
def encoder_layer_vectors(model, repr, device):
    """Run one modality through the M2M encoder and pool each layer over the
    modality positions only (the EOS that merge_modalities appends is dropped)."""
    from multimodalhugs.modules.utils import merge_modalities

    encoder = model.get_backbone_encoder
    n_frames = repr.shape[1]
    mask = torch.ones(repr.shape[:2], dtype=torch.long, device=device)
    inputs_embeds, attn = merge_modalities(
        x=repr,
        padding_mask=mask,
        prompt=None,
        prompt_length_padding_mask=None,
        embeddings_module=encoder.embed_tokens,
        pad_idx=model.pad_token_id,
        eos_idx=model.eos_token_id,
    )
    out = encoder(inputs_embeds=inputs_embeds, attention_mask=attn,
                  output_hidden_states=True, return_dict=True)
    # hidden_states[0] is the encoder input; [1:] are the per-layer outputs.
    return {
        f"enc_{i}": h[:, :n_frames, :].mean(dim=1).squeeze(0).cpu()
        for i, h in enumerate(out.hidden_states[1:], start=1)
    }


@torch.no_grad()
def embed_sample(model, frames, mel, device, with_encoder):
    from multimodalhugs.modules.multimodal_mapper import MultimodalMapper

    frames = frames.unsqueeze(0).to(device)              # [1, T, C, H, W]
    feats = model.feature_extractor(frames)              # [1, T, 512]
    v_mask = torch.ones(feats.shape[:2], dtype=torch.long, device=device)
    video_repr, _ = model.multimodal_mapper(feats, v_mask)
    video = {"mapper": video_repr.mean(dim=1).squeeze(0).cpu()}

    mel = mel.unsqueeze(0).to(device)                    # [1, n_mels, T]
    if mel.shape[-1] < WHISPER_FRAMES:
        mel = F.pad(mel, (0, WHISPER_FRAMES - mel.shape[-1]))
    audio_feats = model.audio_feature_extractor(mel)     # [1, T', 1024]
    a_mask = torch.ones(audio_feats.shape[:2], dtype=torch.long, device=device)
    if isinstance(model.audio_mapper, MultimodalMapper):
        audio_repr, _ = model.audio_mapper(audio_feats, a_mask)
    else:
        audio_repr = model.audio_mapper(audio_feats)
    audio = {"mapper": audio_repr.mean(dim=1).squeeze(0).cpu()}

    if with_encoder:
        video.update(encoder_layer_vectors(model, video_repr, device))
        audio.update(encoder_layer_vectors(model, audio_repr, device))
    return video, audio


def collect_embeddings(model, df, video_proc, audio_proc, device, max_samples, with_encoder):
    video_acc, audio_acc = {}, {}
    n = len(df) if max_samples is None else min(max_samples, len(df))
    for i, row in enumerate(df.itertuples(index=False)):
        if i >= n:
            break
        start = getattr(row, "signal_start", 0) or 0
        end = getattr(row, "signal_end", 0) or 0
        a_start = getattr(row, "audio_start", start) or 0
        a_end = getattr(row, "audio_end", end) or 0
        try:
            frames = video_proc.process_sample({"signal": row.signal, "signal_start": start, "signal_end": end})
            mel = audio_proc.process_sample({"signal": row.signal, "signal_start": a_start, "signal_end": a_end})
        except Exception as exc:
            log.warning("skipping %s: %s", row.signal, exc)
            continue
        video, audio = embed_sample(model, frames, mel, device, with_encoder)
        for k, vec in video.items():
            video_acc.setdefault(k, []).append(vec)
        for k, vec in audio.items():
            audio_acc.setdefault(k, []).append(vec)
        if (i + 1) % 50 == 0:
            log.info("  %d / %d samples", i + 1, n)
    video_emb = {k: torch.stack(v) for k, v in video_acc.items()}
    audio_emb = {k: torch.stack(v) for k, v in audio_acc.items()}
    return video_emb, audio_emb


def ordered_layers(names):
    return sorted(names, key=lambda n: (n != "mapper", int(n.split("_")[1]) if n.startswith("enc_") else 0))


def _rank_metrics(sim, ks):
    """sim[i, j] = similarity of query i to gallery j; the paired item is j == i.

    rank(i) = 1 + #{galleries strictly more similar than the paired one}, so ties
    never penalise the match. Returns R@k, MRR, median and mean rank.
    """
    correct = sim.diagonal().unsqueeze(1)
    rank = (sim > correct).sum(dim=1).float() + 1.0
    out = {f"R@{k}": (rank <= k).float().mean().item() for k in ks}
    out["MRR"] = (1.0 / rank).mean().item()
    out["median_rank"] = rank.median().item()
    out["mean_rank"] = rank.mean().item()
    return out


def retrieval(video, audio, ks):
    sim = F.normalize(video, dim=-1) @ F.normalize(audio, dim=-1).T
    return {"v2a": _rank_metrics(sim, ks), "a2v": _rank_metrics(sim.t(), ks)}


def _avg(d1, d2):
    return {k: (d1[k] + d2[k]) / 2 for k in d1}


def hubness(video, audio, k):
    """How concentrated retrievals are on a few audio embeddings.

    N_k(i) counts how many videos keep audio i among their top-k. A uniform
    space has E[N_k] = k; hubs sit far above it.
    """
    sim = F.normalize(video, dim=-1) @ F.normalize(audio, dim=-1).T
    topk = sim.topk(k, dim=1).indices
    counts = torch.bincount(topk.flatten(), minlength=audio.shape[0]).float()
    return {
        "k_skewness": float(skew(counts.numpy())),
        "hub_rate": (counts > 2 * k).float().mean().item(),
        "antihub_rate": (counts == 0).float().mean().item(),
        "max_N_k": counts.max().item(),
        "mean_N_k": counts.mean().item(),
    }


def relational_similarity(video, audio):
    """RSA (Spearman over pairwise cosine distances) and linear CKA."""
    vn = F.normalize(video, dim=-1)
    an = F.normalize(audio, dim=-1)
    dv = 1 - vn @ vn.T
    da = 1 - an @ an.T
    iu = torch.triu_indices(video.shape[0], video.shape[0], offset=1)
    rho = spearmanr(dv[iu[0], iu[1]].numpy(), da[iu[0], iu[1]].numpy()).correlation

    xc = video - video.mean(dim=0, keepdim=True)
    yc = audio - audio.mean(dim=0, keepdim=True)
    cka = (yc.T @ xc).norm() ** 2 / ((xc.T @ xc).norm() * (yc.T @ yc).norm())
    return {"rsa_spearman": float(rho), "linear_cka": float(cka)}


def relational_baseline(video, audio, n_perm=10, seed=0):
    """Chance level for RSA/CKA: break the pairing by shuffling the audio rows.

    Linear CKA in particular has a positive floor that grows with D/N, so the
    measured value is only meaningful relative to this shuffled baseline.
    """
    g = torch.Generator().manual_seed(seed)
    rsa, cka = [], []
    for _ in range(n_perm):
        perm = torch.randperm(audio.shape[0], generator=g)
        m = relational_similarity(video, audio[perm])
        rsa.append(m["rsa_spearman"])
        cka.append(m["linear_cka"])
    return {"rsa_spearman_chance": float(np.mean(rsa)), "linear_cka_chance": float(np.mean(cka))}


def canonical_correlation(video, audio, n_components):
    from sklearn.cross_decomposition import PLSCanonical

    V, A = video.numpy(), audio.numpy()
    n = min(n_components, V.shape[0] // 5, V.shape[1], A.shape[1])
    if n < 1:
        log.warning("too few samples for CCA (N=%d)", V.shape[0])
        return {}
    pls = PLSCanonical(n_components=n, max_iter=1000)
    pls.fit(V, A)
    Vc, Ac = pls.transform(V, A)
    corrs = [float(np.corrcoef(Vc[:, i], Ac[:, i])[0, 1]) for i in range(n)]
    return {"mean_canonical_corr": float(np.mean(corrs)), "n_components": n, "correlations": corrs}


def report(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))
    for key, val in metrics.items():
        if isinstance(val, list):
            continue
        if isinstance(val, float):
            fmt = f"{val:.1f}" if "rank" in key else f"{val:.4f}"
            print(f"  {key:<22}{fmt}")
        else:
            print(f"  {key:<22}{val}")


def depth_table(video_emb, audio_emb, ks):
    layers = ordered_layers(set(video_emb) & set(audio_emb))
    header = f"{'layer':<8} " + "  ".join(f"R@{k}" for k in ks) + "    MRR    medR     RSA     CKA"
    print("\nAlignment by depth  (retrieval averaged over both directions)")
    print(header)
    print("-" * len(header))
    for name in layers:
        v, a = video_emb[name], audio_emb[name]
        ret = retrieval(v, a, ks)
        r = _avg(ret["v2a"], ret["a2v"])
        rs = relational_similarity(v, a)
        cells = "  ".join(f"{r[f'R@{k}']:.3f}" for k in ks)
        print(f"{name:<8} {cells}  {r['MRR']:.3f}  {r['median_rank']:5.1f}  {rs['rsa_spearman']:+.3f}  {rs['linear_cka']:.3f}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", stream=sys.stdout)

    p = argparse.ArgumentParser(description="Space alignment evaluation")
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--metadata_file", required=True)
    p.add_argument("--video_preprocessor", default=None)
    p.add_argument("--audio_preprocessor", default=None)
    p.add_argument("--skip_frames_stride", type=int, default=3)
    p.add_argument("--retrieval_ks", type=int, nargs="+", default=[1, 5, 10, 100])
    p.add_argument("--n_cca_components", type=int, default=20)
    p.add_argument("--hubness_k", type=int, default=5)
    p.add_argument("--no_encoder", action="store_true", help="only analyse the mapper output")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    video_pp, audio_pp = resolve_preprocessors(args.checkpoint_path, args.video_preprocessor, args.audio_preprocessor)
    log.info("video preprocessor: %s | audio preprocessor: %s", video_pp, audio_pp)

    model = load_model(args.checkpoint_path, device)
    video_proc, audio_proc = build_processors(video_pp, audio_pp, args.skip_frames_stride)

    df = pd.read_csv(args.metadata_file, sep="\t")
    log.info("%d rows in %s", len(df), args.metadata_file)

    video_emb, audio_emb = collect_embeddings(
        model, df, video_proc, audio_proc, device, args.max_samples, with_encoder=not args.no_encoder
    )
    n, d = video_emb["mapper"].shape
    log.info("collected %d pairs, D=%d, layers=%s", n, d, list(video_emb))

    ks = [k for k in args.retrieval_ks if k <= n]
    print(f"\nSpace alignment report  (N={n}, D={d})")

    # Detailed single-point report at the mapper output.
    v0, a0 = video_emb["mapper"], audio_emb["mapper"]
    ret = retrieval(v0, a0, ks)
    report("Retrieval at mapper (video -> audio)", ret["v2a"])
    report("Retrieval at mapper (audio -> video)", ret["a2v"])
    hub = hubness(v0, a0, args.hubness_k)
    report(f"Hubness at mapper (k={args.hubness_k})", hub)
    print("  hubness detected" if hub["k_skewness"] > 1.2 else "  no significant hubness")
    rel = relational_similarity(v0, a0)
    rel.update(relational_baseline(v0, a0))
    report("Relational similarity at mapper (vs shuffled baseline)", rel)
    cca = canonical_correlation(v0, a0, args.n_cca_components)
    if cca:
        report(f"Canonical correlation at mapper ({cca['n_components']} comp)", cca)
        for i, c in enumerate(cca["correlations"], 1):
            print(f"  dim {i:>3}  {c:+.4f}  {'#' * int(max(c, 0) * 20)}")

    # Depth curve across mapper + encoder layers.
    if not args.no_encoder:
        depth_table(video_emb, audio_emb, ks)
    print()


if __name__ == "__main__":
    main()
