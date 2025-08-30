import os, glob, json, argparse
from pathlib import Path
import numpy as np, mne
from mne.preprocessing import Xdawn

def edf_list(session_dir, phase):
    return sorted(glob.glob(os.path.join(session_dir, phase, "**", "*.edf"), recursive=True))

def build_epochs(edf_files, resample=256.0, tmin=-0.2, tmax=0.8, baseline=(-0.2,0.0),
                 notch=(50,), bp=(0.1,15.0), decim=1, max_epochs=None, max_files=None):
    Xs, ys, info_ref = [], [], None
    n_total = 0
    for i, f in enumerate(edf_files):
        if max_files is not None and i >= max_files:
            break
        raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
        if abs(raw.info["sfreq"] - resample) > 1e-6:
            raw.resample(resample)
        raw.rename_channels(lambda s: s.replace("EEG_","") if s.startswith("EEG_") else s)
        if "P08" in raw.ch_names and "PO8" not in raw.ch_names:
            mne.rename_channels(raw.info, {"P08":"PO8"})
        # Build events
        if {"StimulusBegin","StimulusType"}.issubset(set(raw.ch_names)):
            sb = raw.copy().pick(["StimulusBegin"]).get_data().ravel()
            st = raw.copy().pick(["StimulusType"]).get_data().ravel()
            onsets = np.where((sb[:-1] < 0.5) & (sb[1:] >= 0.5))[0] + 1
            if onsets.size == 0:
                continue
            labels = (st[onsets] > 0).astype(np.int32)
            events = np.column_stack([onsets.astype(np.int32),
                                      np.zeros_like(onsets, dtype=np.int32),
                                      labels]).astype(np.int32)
            event_id = {"non":0, "target":1}
        else:
            events, ann = mne.events_from_annotations(raw, verbose=False)
            lower = {k.lower(): v for k, v in ann.items()}
            code_t = lower.get("target"); code_n = lower.get("non") or lower.get("nontarget")
            if code_t is None or code_n is None:
                continue
            keep = np.isin(events[:,2], [code_n, code_t]); events = events[keep].copy()
            events[:,2] = np.where(events[:,2] == code_t, 1, 0)
            events = events.astype(np.int32)
            event_id = {"non":0, "target":1}

        raw.pick_types(eeg=True, eog=False, stim=False, misc=False)
        raw.set_montage("standard_1020", match_case=False, on_missing="ignore")
        if notch:
            raw.notch_filter(freqs=list(notch), picks="eeg")
        raw.filter(bp[0], bp[1], method="iir", picks="eeg")

        ep = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=baseline, preload=True, picks="eeg", decim=decim, verbose=False)
        if len(ep) == 0:
            continue

        if max_epochs is not None and n_total + len(ep) > max_epochs:
            take = max_epochs - n_total
            if take <= 0:
                break
            ep = ep[:take]
        n_total += len(ep)

        if info_ref is None:
            info_ref = ep.info
        Xs.append(ep.get_data())
        ys.append(ep.events[:,2].astype(np.int32))

        if max_epochs is not None and n_total >= max_epochs:
            break

    if not Xs:
        raise RuntimeError("No epochs built.")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0).astype(np.int32)
    return X, y, info_ref, ep.tmin, float(info_ref['sfreq'])

def feats_from_epo(E, n_comp, tmin_epo, sfreq, t0=0.0, t1=0.6, win=0.15, step=0.15):
    start = int((t0 - tmin_epo)*sfreq)
    end   = int((t1 - tmin_epo)*sfreq)
    w     = int(win*sfreq)
    s     = int(step*sfreq)
    starts = list(range(start, max(start, end-w)+1, s))
    F = []
    for e in E:  # (n_comp, n_times)
        vec=[]
        for a in starts:
            b = a + w
            if b > e.shape[1]:
                vec.extend([0.0]*n_comp)
            else:
                vec.extend(e[:, a:b].mean(axis=1))
        F.append(vec)
    return np.asarray(F, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--outdir", default="inputs_xdawn")
    ap.add_argument("--ncomp", type=int, default=3)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=0.6)
    ap.add_argument("--win", type=float, default=0.15)
    ap.add_argument("--step", type=float, default=0.15)
    ap.add_argument("--decim", type=int, default=2)
    ap.add_argument("--max-files-train", type=int, default=None)
    ap.add_argument("--max-files-test",  type=int, default=None)
    ap.add_argument("--max-epochs-train", type=int, default=None)
    ap.add_argument("--max-epochs-test",  type=int, default=None)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    tr_list = edf_list(args.session, "Train")
    te_list = edf_list(args.session, "Test")
    if not tr_list or not te_list:
        raise RuntimeError(f"Không thấy EDF Train/Test trong {args.session}")

    Xtr_raw, ytr, info, tmin_epo, sf = build_epochs(tr_list, decim=args.decim,
        max_epochs=args.max_epochs_train, max_files=args.max_files_train)
    Xte_raw, yte, _, _, _ = build_epochs(te_list, decim=args.decim,
        max_epochs=args.max_epochs_test,  max_files=args.max_files_test)

    # BẮT BUỘC: ép kiểu int cho nhãn
    ytr = ytr.astype(np.int32)
    yte = yte.astype(np.int32)

    # events: ép int cho toàn bộ mảng
    events_tr = np.column_stack([
        np.arange(len(ytr), dtype=np.int32),
        np.zeros(len(ytr), dtype=np.int32),
        ytr
    ]).astype(np.int32)
    event_id = {"non":0, "target":1}

    events_te = np.column_stack([
        np.arange(len(yte), dtype=np.int32),
        np.zeros(len(yte), dtype=np.int32),
        yte
    ]).astype(np.int32)

    ep_tr = mne.EpochsArray(Xtr_raw, info=info, events=events_tr, event_id=event_id, tmin=tmin_epo, verbose=False)
    ep_te = mne.EpochsArray(Xte_raw, info=info, events=events_te, event_id=event_id, tmin=tmin_epo, verbose=False)

    xd = Xdawn(n_components=args.ncomp, reg=0.1)
    xd.fit(ep_tr)
    Etr = xd.transform(ep_tr)
    Ete = xd.transform(ep_te)

    Ftr = feats_from_epo(Etr, args.ncomp, tmin_epo, sf, args.t0, args.t1, args.win, args.step)
    Fte = feats_from_epo(Ete, args.ncomp, tmin_epo, sf, args.t0, args.t1, args.win, args.step)

    np.savez_compressed(out/"train_features.npz", X=Ftr, y=ytr)
    np.savez_compressed(out/"test_features.npz",  X=Fte, y=yte)
    meta = {"sfreq": sf, "tmin": tmin_epo, "ncomp": args.ncomp, "t0": args.t0, "t1": args.t1,
            "win": args.win, "step": args.step, "decim": args.decim}
    (out/"features_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved:", out/"train_features.npz", out/"test_features.npz")
    print("Train:", Ftr.shape, " Test:", Fte.shape, " Pos-rate:", ytr.mean(), yte.mean())

if __name__ == "__main__":
    main()

