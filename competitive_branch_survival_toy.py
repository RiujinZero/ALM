import numpy as np

SEED = 31
rng = np.random.default_rng(SEED)


def make_events(n_mem=80, cue_dim=256, cue_k=28, n_fam=8, shared_k=12):
    X = np.zeros((n_mem, cue_dim), dtype=np.float32)
    cores = []
    for _family in range(n_fam):
        cores.append(rng.choice(cue_dim, shared_k, replace=False))
    for m in range(n_mem):
        X[m, cores[m % n_fam]] = 1
        avail = np.where(X[m] == 0)[0]
        X[m, rng.choice(avail, cue_k - shared_k, replace=False)] = 1
    return X


def branch_response(branch, x, need=2):
    # local NMDA-like coincidence: require >=need members, graded above threshold
    hits = x[branch].sum()
    return 0.0 if hits < need else hits / len(branch)


def build_branches(X, B=10, candidates=80, branch_k=3, competitive=True, decay=0.30):
    n = len(X)
    branches = [[] for _ in range(n)]
    weights = [np.zeros(0) for _ in range(n)]
    for m in range(n):
        active = np.where(X[m] > 0)[0]
        cand = []
        seen = set()
        while len(cand) < candidates:
            tup = tuple(sorted(rng.choice(active, branch_k, replace=False)))
            if tup not in seen:
                seen.add(tup)
                cand.append(np.array(tup, dtype=int))
        if competitive and m > 0:
            # Local survival score: own coincidence minus firing on previously
            # active alternatives.
            prev = X[:m]
            false_rates = []
            for br in cand:
                false_rates.append(
                    np.mean([branch_response(br, p) >= 2 / 3 for p in prev])
                )
            order = np.argsort(false_rates)
            chosen = [cand[i] for i in order[:B]]
            w = np.exp(-4 * np.array(false_rates)[order[:B]])
        else:
            chosen = cand[:B]
            w = np.ones(B)
        branches[m] = chosen
        weights[m] = w.astype(np.float32)
        if competitive:
            # New experiences depress old branches that remain nondiscriminative.
            for j in range(m):
                if len(branches[j]) == 0:
                    continue
                for k, br in enumerate(branches[j]):
                    if branch_response(br, X[m]) >= 2 / 3:
                        weights[j][k] *= 1 - decay
                # eliminate nearly dead branches; no oracle replacement.
                keep = weights[j] > 0.12
                branches[j] = [
                    br for br, kp in zip(branches[j], keep, strict=True) if kp
                ]
                weights[j] = weights[j][keep]
    return branches, weights


def apical_support(x, branches, weights):
    out = np.zeros(len(branches))
    for m, (bs, ws) in enumerate(zip(branches, weights, strict=True)):
        if len(bs) == 0:
            continue
        rs = np.array([branch_response(br, x) for br in bs])
        out[m] = np.dot(ws, rs) / (ws.sum() + 1e-9)
    return out


def partial(x, keep=0.35):
    y = np.zeros_like(x)
    a = np.where(x > 0)[0]
    y[a[rng.random(len(a)) < keep]] = 1
    return y


def blend(a, b, keep=0.35):
    return np.maximum(partial(a, keep), partial(b, keep))


def dynamics(q, steps=50):
    # Assembly-level basal completion + local commitment.  Conflict is generated
    # by pairwise suprathreshold apical co-activation; no winner/ranking is used.
    h = np.clip(q, 0, 1)
    c = np.zeros_like(q)
    conflict = 0.0
    for _ in range(steps):
        pair = (q.sum() ** 2 - np.dot(q, q)) / 2.0
        conflict = 0.85 * conflict + 0.15 * pair
        h_target = 1 / (
            1 + np.exp(-np.clip(-2.5 + 12 * q + 3.0 * h - 1.0 * (h.sum() - h), -30, 30))
        )
        h = 0.7 * h + 0.3 * h_target
        lateral = c.sum() - c
        coincidence = h * q
        drive = (
            -4.0 + 14 * coincidence + 7 * q + 2.3 * c - 2.4 * lateral - 10.0 * conflict
        )
        c_target = 1 / (1 + np.exp(-np.clip(drive, -30, 30)))
        c = 0.76 * c + 0.24 * c_target
    return h, c, conflict


def eval_system(n_mem, competitive, trials=400, keep=0.35):
    X = make_events(n_mem=n_mem)
    branches, weights = build_branches(X, competitive=competitive)
    commits = correct = 0
    eligible = []
    amb_commit = 0
    for _ in range(trials):
        m = int(rng.integers(n_mem))
        y = partial(X[m], keep)
        q = apical_support(y, branches, weights)
        eligible.append(np.sum(q > 0.15))
        h, c, _ = dynamics(q)
        if c.max() > 0.70:
            commits += 1
            correct += int(np.argmax(c) == m)
    for _ in range(trials):
        a, b = rng.choice(n_mem, 2, replace=False)
        q = apical_support(blend(X[a], X[b], keep), branches, weights)
        h, c, _ = dynamics(q)
        amb_commit += int(c.max() > 0.70)
    return dict(
        n_mem=n_mem,
        competitive=competitive,
        commit_rate=commits / trials,
        precision=correct / max(commits, 1),
        ambiguous_commit=amb_commit / trials,
        mean_eligible=float(np.mean(eligible)),
        p95_eligible=float(np.quantile(eligible, 0.95)),
        mean_surviving_branches=float(np.mean([len(b) for b in branches])),
    )


if __name__ == "__main__":
    for n in (12, 32, 64, 96):
        for comp in (False, True):
            print(eval_system(n, comp))
