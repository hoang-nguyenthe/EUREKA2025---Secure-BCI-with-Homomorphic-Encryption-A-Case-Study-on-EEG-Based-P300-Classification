import os, glob, json, argparse
from pathlib import Path
import numpy as np, mne

BASE_CHS = ["F3","Fz","F4","C3","Cz","C4","P3","Pz","P4","PO7","PO8","Oz","Fp1","Fp2","O1","O2"]

def edf_list(session_dir, phase):
    return sorted(glob.glob(os.path.join(session_dir, phase, "**", "*.edf"), recursive=True))

def epochs_from_files(edf_files, resample=256.0, tmin=-0.2, tmax=0.8, baseline=(-0.2,0.0),
                      notch=(50,), bp=(0.1,15.0), decim=1, max_epochs=None, max_files=None):
    Xs, ys, info_ref = [], [], None
    n_ep_total = 0
    for fi, f in enumerate(edf_files):
        if max_files is not None and fi >= max_files: break
        raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
        if abs(raw.info["sfreq"]-resample) > 1e-6:
            raw.resample(resample)
        raw.rename_channels(lambda s: s.replace("EEG_","") if s.startswith("EEG_") else s)
        if "P08" in raw.ch_names and "PO8" not in raw.ch_names:
            mne.rename_channels(raw.info, {"P08":"PO8"})
        # Events
        if {"StimulusBegin","StimulusType"}.issubset(set(raw.ch_names)):
            sb = raw.copy().pick(["StimulusBegin"]).get_data().ravel()
            st = raw.copy().pick(["StimulusType"]).get_data().ravel()
            onsets = np.where((sb[:-1]<0.5)&(sb[1:]>=0.5))[0]+1
            if onsets.size==0: continue
            labels = (st[onsets]>0).astype(int)
            events = np.column_stack([onsets, np.zeros_like(onsets), labels]).astype(int)
            event_id={"non":0,"target":1}
        else:
            events, ann = mne.events_from_annotations(raw, verbose=False)
            lower = {k.lower(): v for k,v in ann.items()}
            code_t = lower.get("target"); code_n = lower.get("non") or lower.get("nontarget")
            if code_t is None or code_n is None: continue
            keep = np.isin(events[:,2],[code_n,code_t]); events = events[keep]
            events[:,2] = np.where(events[:,2]==code_t,1,0); event_id={"non":0,"target":1}
        raw.pick_types(eeg=True, eog=False, stim=False, misc=False)
        raw.set_montage("standard_1020", match_case=False, on_missing="ignore")
        if notch: raw.notch_filter(freqs=list(notch), picks="eeg")
        raw.filter(bp[0], bp[1], method="iir", picks="eeg")
        ep = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=baseline, preload=True, picks="eeg", decim=decim, verbose=False)
        if len(ep)==0: continue

        # Nếu cần cắt bớt epoch để giữ RAM thấp
        if max_epochs is not None and n_ep_total + len(ep) > max_epochs:
            take = max_epochs - n_ep_total
            if take <= 0: break
            ep = ep[:take]
        n_ep_total += len(ep)

        if info_ref is None: info_ref = ep.info
        Xs.append(ep.get_data()); ys.append(ep.events[:,2])

        if max_epochs is not None and n_ep_total >= max_epochs:
            break

    if not Xs: raise RuntimeError("No epochs built.")
    X = np.concatenate(Xs, axis=0); y = np.concatenate(ys, axis=0).astype(np.int32)
    return X, y, info_ref

def extract_64(X, info, tmin=-0.2, sfreq=256.0):
    start_idx = int((0.0 - tmin) * sfreq)
    w = int(0.15 * sfreq)
    idx_windows = [(start_idx + i*w, start_idx + (i+1)*w) for i in range(4)]
    ch_to_idx = {ch:i for i,ch in enumerate(info.ch_names)}
    feats = []
    for e in X:
        vec = []
        for ch in BASE_CHS:
            ci = ch_to_idx.get(ch, None)
            for a,b in idx_windows:
                if ci is None or b>e.shape[1]: vec.append(0.0)
                else:                           vec.append(float(e[ci, a:b].mean()))
        feats.append(vec)
    F = np.asarray(feats, dtype=np.float32)
    assert F.shape[1]==64, f"Expect 64 features, got {F.shape[1]}"
    return F

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="VD: /work/bigP3BCI-data/StudyX/X_YY/SEZZZ")
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--decim", type=int, default=1, help="decimate factor (1=không giảm; 2=~128Hz, 4=~64Hz)")
    ap.add_argument("--max-files-train", type=int, default=None)
    ap.add_argument("--max-files-test",  type=int, default=None)
    ap.add_argument("--max-epochs-train", type=int, default=None)
    ap.add_argument("--max-epochs-test",  type=int, default=None)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    tr_list = edf_list(args.session, "Train")
    te_list = edf_list(args.session, "Test")
    if not tr_list or not te_list:
        raise RuntimeError(f"Không tìm thấy EDF Train/Test trong {args.session}")

    Xtr_e, ytr, info = epochs_from_files(tr_list, decim=args.decim,
        max_epochs=args.max_epochs_train, max_files=args.max_files_train)
    Xte_e, yte, _    = epochs_from_files(te_list,  decim=args.decim,
        max_epochs=args.max_epochs_test,  max_files=args.max_files_test)

    Ftr = extract_64(Xtr_e, info, tmin=-0.2, sfreq=float(info['sfreq']))
    Fte = extract_64(Xte_e, info, tmin=-0.2, sfreq=float(info['sfreq']))

    np.savez_compressed(out/"train_features.npz", X=Ftr, y=ytr)
    np.savez_compressed(out/"test_features.npz",  X=Fte, y=yte)
    meta = {"sfreq": float(info["sfreq"]), "ch_names": info.ch_names, "decim": args.decim}
    (out/"features_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:", out/"train_features.npz", out/"test_features.npz")
    print("Train:", Ftr.shape, " Test:", Fte.shape, " Pos-rate:", ytr.mean(), yte.mean())
    print(f"(decim={args.decim}, max_files_train={args.max_files_train}, max_epochs_train={args.max_epochs_train})")

if __name__ == "__main__":
    main()

