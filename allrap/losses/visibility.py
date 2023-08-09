def weak_negative_correlation(v, z):
    v=v.flatten()
    z=z.flatten()
    v=v-v.mean()
    z=z-z.mean()
    cs=torch.nn.functional.cosine_similarity(v,z,dim=0)
    return cs.clamp(min=-0.05)
