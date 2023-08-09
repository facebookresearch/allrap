import torch
import tuples
import data.sup3d

dataloader = data.sup3d.loaders()

print()
1/0


log = logging.getLogger(__name__)
tmp_dir = "/scratch/benjamingraham/tmp_dir"
if not os.path.exists(tmp_dir):
    tmp_dir = f'/scratch/slurm_tmpdir/{os.environ["SLURM_JOB_ID"]}'

def printl(*args):
    log.info(" ".join(map(str, args)))

printl(cfg)
printl("Working directory:", os.getcwd())
printl("tmp_dir", tmp_dir)

CFG = ExperimentConfig(
    cfg_file="/private/home/benjamingraham/repos/c3dpo_nrsfm/cfgs/up3d_79kp.yaml"
).cfg
CFG.batch_size = cfg.batch_size

dset_train, dset_val, dset_test = dataset_zoo(**CFG.DATASET)
print(len(dset_train), len(dset_val), len(dset_test))  # 171090 15000 15000

trainloader = torch.utils.data.DataLoader(
    dset_train,
    num_workers=CFG.num_workers,
    pin_memory=True,
    batch_size=CFG.batch_size,
    shuffle=True,
    drop_last=True,
)
valloader = torch.utils.data.DataLoader(
    dset_val,
    num_workers=CFG.num_workers,
    pin_memory=True,
    batch_size=CFG.batch_size,
    shuffle=False,
)
testloader = torch.utils.data.DataLoader(
    dset_test,
    num_workers=CFG.num_workers,
    pin_memory=True,
    batch_size=CFG.batch_size,
    shuffle=False,
)
eval_script, cache_vars, eval_vars = eval_zoo(CFG.DATASET.dataset_name)

dtype = getattr(torch, cfg.dtype)
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

tuple_losses = torch.nn.ModuleList(
    [
        # tuples.TupleAlignedCoordLoss(2,32,CFG.MODEL.n_keypoints),
        # tuples.TupleAlignedCoordLoss2(tuples.random_k_tuples(5,48,CFG.MODEL.n_keypoints),alpha=1)
    ]
)

if cfg.transformer:
    net = keypoint_transformer.KeypointTransformer()
else:
    # net=tuples.PoseNet(CFG.MODEL.n_keypoints,activ=cfg.activ,n=cfg.n_hidden,hidden_layers=cfg.layers,bn=cfg.bn)
    # net=MLPNonMixerPoseNet(hidden_layers=12)
    net = MLPMixerPoseNet(
        dropout=0, hidden_layers=32, dim=32, activation=torch.nn.ReLU
    )
    # net=Transformer()
print(net)
combo = torch.nn.Sequential(net, tuple_losses).to(dtype).to(cfg.device)

optim = torch.optim.Adam(
    [
        {"params": net.parameters(), "lr": 1e-3},
    ]
)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.1 ** (1 / 300))

filename = "state.pth"
if os.path.exists(filename):
    printl(filename)
    try:
        epoch, state = torch.load(filename)
        # for _ in range(epoch):
        #    scheduler.step()
    except:
        epoch, state = torch.load(filename + "_")
        # for _ in range(epoch):
        #    scheduler.step()
    combo.load_state_dict(state, strict=True)
    printl(epoch)
else:
    epoch = 0
for epoch in range(epoch, 300):
    combo.train()
    total_loss = []
    for s in trainloader:
        with torch.no_grad():
            pose_2d = (
                torch.cat(
                    [s["kp_loc"].transpose(-2, -1), s["kp_vis"][:, :, None]], 2
                )
                .to(dtype)
                .to(cfg.device)
            )
            # random 2d rotation
            R = torch.rand(len(pose_2d)) * 2 * np.pi
            R = torch.stack([R.cos(), R.sin(), -R.sin(), R.cos()], 1).reshape(
                -1, 2, 2
            )
            R = R.to(dtype).to(cfg.device)
            pose_2d[:, :, :2] = pose_2d[:, :, :2] @ R
        optim.zero_grad()
        pred_pose_3d = net(pose_2d.to(dtype))

        z0 = pose_2d[:, :, 2].reshape(-1)
        z1 = pred_pose_3d[:, :, 2].reshape(-1)
        loss = (
            torch.nn.functional.cosine_similarity(         z0 - z0.mean(), z1 - z1.mean(), dim=0   ).clamp(min=-0.05)
            #+ tuples.random_tuple_aligned_coord_loss(pred_pose_3d, 10, 32, False)
            #+ tuples.nearby_tuple_aligned_coord_loss(pred_pose_3d, 10, 8, False)
            #+ tuples.nearby_tuple_aligned_coord_loss(pred_pose_3d, 10, 16, False)
            + tuples.nearby_tuple_aligned_coord_loss(pred_pose_3d, 10, 32, False)
            )
        timm.utils.adaptive_clip_grad(combo.parameters(), clip_factor=0.1)
        loss.backward()
        optim.step()
        total_loss.append(loss.item())
    # scheduler.step()

    torch.save([epoch + 1, combo.state_dict()], filename + "_")
    shutil.copyfile(filename + "_", filename)

    if epoch > 5 and epoch % 5 < 4:
        continue
    combo.eval()
    gt = []
    pred = []
    visi = []
    with torch.no_grad():
        for s in testloader:
            pose_2d = torch.cat(
                [s["kp_loc"].transpose(-2, -1), s["kp_vis"][:, :, None]], 2
            ).to(cfg.device)
            pred_pose_3d = net(pose_2d.to(dtype))
            gt.append(s["kp_loc_3d"])
            visi.append(s["kp_vis"])
            pred.append(pred_pose_3d.transpose(-2, -1).cpu().detach())
        gt = torch.cat(gt)
        pred = torch.cat(pred)
        visi = torch.cat(visi)
        pred[:, 2, :] -= pred[:, 2, :].mean(1, keepdim=True)
        results, _ = data.sup3d.eval_up3d_79kp(
            {"kp_loc_3d": gt, "shape_image_coord": pred}
        )
        print(epoch, results)

my_app()

