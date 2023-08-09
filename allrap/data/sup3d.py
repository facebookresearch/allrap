import torch, os, json, urllib.request

# https://github.com/facebookresearch/c3dpo_nrsfm/blob/main/dataset/dataset_configs.py

DATASET_URL = {
    "train": "https://dl.fbaipublicfiles.com/c3dpo_nrsfm/up3d_79kp_train.json",
    "test": "https://dl.fbaipublicfiles.com/c3dpo_nrsfm/up3d_79kp_test.json",
}

DATASET_MD5 = {
    "train": "fde2aee038ecd0f145181559eff59c9f",
    "test": "7d8bf3405ec085394e9257440e8bcb18",
}


def loaders(batch_size=64):
    root_dir = os.path.dirname(os.path.realpath(__file__))
    dl = {}
    for split, url in DATASET_URL.items():
        f = os.path.join(root_dir, url.split("/")[-1])
        if not os.path.exists(f):
            print(f"Downloading {url} to {f} ...")
            urllib.request.urlretrieve(url, f)
        print(f"Loading {f} ...")
        dset = json.load(open(f, "r"))["data"]
        dl[split] = torch.utils.data.DataLoader(
            dset["train"],
            num_workers=8,
            pin_memory=True,
            batch_size=batch_size,
            shuffle=split == "train",
            drop_last=split == "train",
        )
    return dl

def loaders(batch_size=64):
    print('xxxxxxxxxxxxxxxxxxxxxxxxxdebug debug debug debug xxxxxxxxxxxxxxxxxxxxxxx')
    root_dir = os.path.dirname(os.path.realpath(__file__))
    f = os.path.join(root_dir, 'up3d_79kp_test.json')
    dset = json.load(open(f, "r"))["data"]
    dl={}
    for split in ['train','test']:
        dl[split] = torch.utils.data.DataLoader(
            dset,
            num_workers=8,
            pin_memory=True,
            batch_size=batch_size,
            shuffle=split == "train",
            drop_last=split == "train",
        )
    return dl

# https://github.com/facebookresearch/c3dpo_nrsfm/blob/main/dataset/eval_zoo.py

def calc_dist_err(gt, pred, scale=1.,
                  fix_mean_depth=False,
                  get_best_scale=False,
                  mask=None):

    assert pred.shape[1] == 3
    assert gt.shape[1] == 3
    assert pred.shape[0] == gt.shape[0]
    assert pred.shape[2] == gt.shape[2]

    if fix_mean_depth:
        # print('setting mean depth = 0')
        pred = set_mean_depth_to_0(pred, mask=mask)
        gt = set_mean_depth_to_0(gt,   mask=mask)

    if get_best_scale:
        argmin_scale = compute_best_scale(pred, gt, v=mask)
        pred = pred.copy() * argmin_scale[:, None, None]

    df = pred - gt
    errs = np_safe_sqrt((df*df).sum(1))

    if True:
        errs_ = np.sqrt((df*df).sum(1))
        df__ = np.max(np.abs(errs-errs_))
        assert df__ <= 1e-5
        # print('err diff = %1.2e' % df__)

    if mask is not None:
        assert mask.shape[0] == pred.shape[0]
        assert mask.shape[1] == pred.shape[2]
        assert len(mask.shape) == 2
        errs = (mask*errs).sum(1) / mask.sum(1)
    else:
        errs = errs.mean(1)

    errs = errs * scale
    return errs

def calc_3d_errs(
    pred, gt, fix_mean_depth=False, get_best_scale=False, scale=float(1), mask=None
):

    pred_flip = np.copy(pred)
    pred_flip[:, 2, :] = -pred_flip[:, 2, :]

    pairs_compare = {"EVAL_MPJPE_orig": pred, "EVAL_MPJPE_flip": pred_flip}

    results = {}
    for metric, pred_compare in pairs_compare.items():
        results[metric] = calc_dist_err(
            gt,
            pred_compare,
            fix_mean_depth=fix_mean_depth,
            get_best_scale=get_best_scale,
            scale=scale,
            mask=mask,
        )

    results["EVAL_MPJPE_best"] = np.minimum(
        results["EVAL_MPJPE_orig"], results["EVAL_MPJPE_flip"]
    )

    results["EVAL_stress"] = calc_stress_err(gt, pred, mask=mask, scale=scale)

    return results


def eval_up3d_79kp(cached_preds, eval_vars=['EVAL_MPJPE_orig', 'EVAL_MPJPE_best', 'EVAL_stress'
], N_ENTRIES=15000):

    print("UP3D evaluation ... (tgt n entries = %d)" % N_ENTRIES)

    gt = np.array(cached_preds["kp_loc_3d"])
    pred = np.array(cached_preds["shape_image_coord"])

    for arr in (gt, pred):
        assert len(arr) == N_ENTRIES, "wrong n of predictions!"

    results = calc_3d_errs(pred, gt, fix_mean_depth=True)

    metrics = list(results.keys())

    # check that eval_vars are all evaluated
    if eval_vars is not None:
        for m in eval_vars:
            assert m in metrics, "missing metric %s!" % m

    all_avg_results = {}
    for metric in metrics:
        all_avg_results[metric] = float(np.array(results[metric]).mean())
        print("%20s: %20s" % (metric, "%1.4f" % all_avg_results[metric]))

    aux_out = {}
    aux_out["per_sample_err"] = results

    return all_avg_results, aux_out
