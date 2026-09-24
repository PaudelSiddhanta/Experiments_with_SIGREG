import torch
import torch.nn.functional as F


EPS = 1e-8


def _center(features):
    return features - features.mean(dim=0, keepdim=True)


def _singular_values(features):
    """
    Singular values of mean-centered feature matrix.

    features: [N, D]
    """
    x = _center(features)

    return torch.linalg.svdvals(x)


# =========================================================
# Simple feature statistics
# =========================================================

def feature_std_metrics(
    features,
    zero_threshold=1e-4,
):
    """
    Per-feature standard deviation across samples.

    Collapse makes these values approach zero.
    """

    std = features.std(
        dim=0,
        unbiased=False,
    )

    return {
        "mean_feature_std":
            std.mean().item(),

        "min_feature_std":
            std.min().item(),

        "near_zero_std_ratio":
            (std < zero_threshold)
            .float()
            .mean()
            .item(),
    }


def mean_feature_norm(features):
    """
    Average L2 norm of representation vectors.
    """

    return (
        features.norm(dim=1)
        .mean()
        .item()
    )


# =========================================================
# Rank metrics
# =========================================================

def total_feature_variance(features):
    """
    Sum of per-dimension variance across samples.
    """

    variance = features.var(
        dim=0,
        unbiased=False,
    )

    return variance.sum().item()


def constant_collapse(
    features,
    threshold=1e-8,
):
    """
    True when total across-sample variance is
    essentially zero.
    """

    return (
        total_feature_variance(features)
        < threshold
    )


def rank_99(
    features,
    threshold=0.99,
):
    """
    Smallest k whose top-k singular directions explain
    at least `threshold` fraction of total variance.

    Returns 0 if the centered representation has
    essentially zero total variance.
    """

    s = _singular_values(features)

    energy = s.square()
    total = energy.sum()

    if total.item() < EPS:
        return 0

    cumulative = torch.cumsum(
        energy,
        dim=0,
    ) / total

    indices = torch.nonzero(
        cumulative >= threshold
    )

    if len(indices) == 0:
        return int(len(s))

    return int(indices[0].item() + 1)


def effective_rank(features):
    """
    Entropy effective rank:

        exp(-sum p_i log p_i)

    Returns 0 for a constant representation with
    zero centered variance.
    """

    s = _singular_values(features)

    energy = s.square()
    total = energy.sum()

    if total.item() < EPS:
        return 0.0

    p = energy / total

    entropy = -torch.sum(
        p * torch.log(
            p.clamp_min(EPS)
        )
    )

    return torch.exp(entropy).item()


def stable_rank(features):
    """
    Stable rank:

        ||H||_F^2 / ||H||_2^2

    for mean-centered H.

    Equivalent to:

        sum_i s_i^2 / s_1^2
    """

    s = _singular_values(features)

    if len(s) == 0:
        return 0.0

    numerator = s.square().sum()

    denominator = (
        s[0].square().clamp_min(EPS)
    )

    return (
        numerator / denominator
    ).item()


# =========================================================
# NRC1-style PCA residual
# =========================================================

def nrc1_residual(
    features,
    k,
):
    """
    Residual after projecting normalized centered
    representations onto their top-k PCA subspace.

    Near 0:
        representation almost entirely fits into
        a k-dimensional subspace.

    Large:
        substantial representation energy remains
        outside that subspace.
    """

    h = _center(features)

    norms = h.norm(
        dim=1,
        keepdim=True,
    ).clamp_min(EPS)

    h_normalized = h / norms

    # Vh shape:
    # [min(N,D), D]
    _, _, vh = torch.linalg.svd(
        h,
        full_matrices=False,
    )

    k = min(
        k,
        vh.shape[0],
    )

    pcs = vh[:k]

    # Instead of explicitly making P = PCs^T PCs:
    #
    # projection coefficients:
    #     [N,D] @ [D,k]
    #
    coefficients = (
        h_normalized @ pcs.T
    )

    # reconstruct projection
    projected = (
        coefficients @ pcs
    )

    residual = (
        projected - h_normalized
    ).square().sum(dim=1)

    return residual.mean().item()


def nrc1_curve(
    features,
    k_values=(1, 2, 5, 10, 20, 50, 100, 150, 200, 256, 300, 400, 500),
):
    """
    NRC1 residual for several values of k.

    Performs ONE SVD and reuses the same principal
    components for all k values.
    """

    h = _center(features)

    norms = h.norm(
        dim=1,
        keepdim=True,
    ).clamp_min(EPS)

    h_normalized = h / norms

    # One SVD only.
    _, _, vh = torch.linalg.svd(
        h,
        full_matrices=False,
    )

    max_rank = vh.shape[0]

    results = {}

    for k in k_values:

        if k > max_rank:
            continue

        pcs = vh[:k]

        coefficients = (
            h_normalized @ pcs.T
        )

        projected = (
            coefficients @ pcs
        )

        residual = (
            projected - h_normalized
        ).square().sum(dim=1)

        results[f"nrc1_k{k}"] = (
            residual.mean().item()
        )

    return results


def covariance_offdiag_ratio(features):
    """
    Ratio:

        ||C - diag(C)||_F / ||C||_F

    where C is feature covariance.

    0 means covariance is perfectly diagonal.
    Larger values mean stronger redundancy /
    correlation between feature dimensions.
    """

    x = _center(features)

    n = x.shape[0]

    covariance = (
        x.T @ x
    ) / max(n - 1, 1)

    diagonal = torch.diag(
        torch.diag(covariance)
    )

    offdiag = covariance - diagonal

    numerator = torch.linalg.norm(
        offdiag,
        ord="fro",
    )

    denominator = torch.linalg.norm(
        covariance,
        ord="fro",
    ).clamp_min(EPS)

    return (
        numerator / denominator
    ).item()


# =========================================================
# Pairwise similarity
# =========================================================

def mean_pairwise_cosine(
    features,
    num_pairs=20000,
    seed=0,
):
    """
    Mean cosine similarity between randomly chosen
    pairs of DIFFERENT samples.

    Near 1 may indicate directional collapse.

    Sampling pairs avoids constructing an NxN matrix.
    """

    x = F.normalize(
        features,
        dim=1,
    )

    n = x.shape[0]

    if n < 2:
        return 1.0

    generator = torch.Generator(
        device=x.device
    )

    generator.manual_seed(seed)

    i = torch.randint(
        0,
        n,
        (num_pairs,),
        generator=generator,
        device=x.device,
    )

    offset = torch.randint(
        1,
        n,
        (num_pairs,),
        generator=generator,
        device=x.device,
    )

    j = (i + offset) % n

    similarities = (
        x[i] * x[j]
    ).sum(dim=1)

    return (
        similarities.mean().item()
    )


# =========================================================
# Sparsity / coordinate usage
# =========================================================

def feature_entropy(
    features,
):
    """
    Shannon entropy of normalized absolute coordinate
    magnitudes within each sample.

    Higher:
        feature magnitude spread across more coordinates.

    Lower:
        feature concentrated in fewer coordinates.

    Note:
        basis-dependent, so use as a secondary diagnostic.
    """

    x = features.abs()

    p = x / (
        x.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(EPS)
    )

    entropy = -torch.sum(
        p * torch.log(
            p.clamp_min(EPS)
        ),
        dim=1,
    )

    return entropy.mean().item()


def gini_sparsity(features):
    """
    Gini coefficient of absolute feature magnitudes.

    0:
        magnitudes distributed uniformly.

    Close to 1:
        representation concentrated in very few
        coordinates.

    This is also basis-dependent.
    """

    x = features.abs()

    x_sorted, _ = torch.sort(
        x,
        dim=1,
    )

    n_features = x.shape[1]

    indices = torch.arange(
        1,
        n_features + 1,
        device=x.device,
        dtype=x.dtype,
    )

    row_sum = (
        x_sorted.sum(
            dim=1
        ).clamp_min(EPS)
    )

    weighted = (
        x_sorted
        * (
            n_features
            - indices
            + 0.5
        )
        / n_features
    ).sum(dim=1)

    gini = (
        1.0
        - 2.0
        * weighted
        / row_sum
    )

    return gini.mean().item()


# =========================================================
# Alignment between augmentations
# =========================================================

def alignment(
    view1_features,
    view2_features,
):
    """
    Mean squared distance between normalized
    representations of two augmented views of
    the same samples.

    Lower = stronger augmentation invariance.

    Collapse also gives excellent alignment, so this
    metric MUST be interpreted alongside diversity/rank.
    """

    x = F.normalize(
        view1_features,
        dim=1,
    )

    y = F.normalize(
        view2_features,
        dim=1,
    )

    return (
        (x - y)
        .square()
        .sum(dim=1)
        .mean()
        .item()
    )


# =========================================================
# Main aggregation function
# =========================================================

def compute_representation_metrics(
    features,
    nrc_k_values=(1, 2, 5, 10, 20, 50, 100, 150, 200, 256, 300, 400, 500),
    zero_std_threshold=1e-4,
    cosine_pairs=20000,
):
    """
    Compute the core metrics for a representation matrix.

    features:
        Tensor of shape [N, D]
    """

    features = (
        features.detach()
        .float()
    )

    results = {}

    results["total_feature_variance"] = (
        total_feature_variance(features)
    )

    results["constant_collapse"] = (
        constant_collapse(features)
    )

    results.update(
        feature_std_metrics(
            features,
            zero_threshold=zero_std_threshold,
        )
    )

    results["mean_feature_norm"] = (
        mean_feature_norm(features)
    )

    results["rank_99"] = (
        rank_99(features)
    )

    results["effective_rank"] = (
        effective_rank(features)
    )

    results["stable_rank"] = (
        stable_rank(features)
    )

    results[
        "covariance_offdiag_ratio"
    ] = covariance_offdiag_ratio(
        features
    )

    results[
        "mean_pairwise_cosine"
    ] = mean_pairwise_cosine(
        features,
        num_pairs=cosine_pairs,
    )

    results["feature_entropy"] = (
        feature_entropy(features)
    )

    results["gini_sparsity"] = (
        gini_sparsity(features)
    )

    results.update(
        nrc1_curve(
            features,
            k_values=nrc_k_values,
        )
    )

    return results
