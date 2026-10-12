import numpy as np

SEED = 23
rng = np.random.default_rng(SEED)


def make_system(
    n_mem=12,
    cue_dim=144,
    asm_k=12,
    cue_k=20,
    shared_k=7,
    n_fam=4,
    conn_p=0.35,
    n_branches=5,
):
    asm_dim = n_mem * asm_k
    X = np.zeros((n_mem, cue_dim))
    fam_shared, avail = [], list(range(cue_dim))
    for _ in range(n_fam):
        s = rng.choice(avail, shared_k, replace=False)
        fam_shared.append(s)
        avail = [i for i in avail if i not in set(s)]
    H = np.zeros((n_mem, asm_dim))
    for m in range(n_mem):
        X[m, fam_shared[m % n_fam]] = 1
        remain = np.where(X[m] == 0)[0]
        X[m, rng.choice(remain, cue_k - shared_k, replace=False)] = 1
        H[m, m * asm_k : (m + 1) * asm_k] = 1
    W_cue = np.zeros((asm_dim, cue_dim))
    branches = np.zeros((n_mem, n_branches, cue_dim))
    for m in range(n_mem):
        ci, hi = np.where(X[m] > 0)[0], np.where(H[m] > 0)[0]
        # Each apical branch recognizes a conjunction of features belonging to
        # this event.  Branches receive cortical input directly; recurrent
        # completion cannot activate them.
        for branch, features in enumerate(
            np.array_split(rng.permutation(ci), n_branches)
        ):
            branches[m, branch, features] = 1
        mask = rng.random((asm_k, len(ci))) < conn_p
        for row in range(asm_k):
            if not mask[row].any():
                mask[row, rng.integers(len(ci))] = True
        W_cue[np.ix_(hi, ci)] = mask
    W_rec = H.T @ H
    np.fill_diagonal(W_rec, 0)
    W_rec /= asm_k - 1
    associated = (H.sum(0) > 0).astype(float)
    W_inh = np.outer(associated, associated)
    W_inh[(H.T @ H) > 0] = 0
    np.fill_diagonal(W_inh, 0)
    W_inh /= asm_k
    receptors = np.arange(n_mem) % 6
    W_return = np.zeros((6, asm_dim))
    for m, r in enumerate(receptors):
        W_return[r] += H[m]
    return X, H, receptors, W_cue, W_rec, W_inh, W_return, branches


X, H, receptors, W_cue, W_rec, W_inh, W_return, branches = make_system()


def partial_cue(x, keep=0.25, false_on=0.002):
    y = np.zeros_like(x)
    active = np.where(x > 0)[0]
    y[active[rng.random(len(active)) < keep]] = 1
    inactive = np.where(x == 0)[0]
    y[inactive[rng.random(len(inactive)) < false_on]] = 1
    return y


def blended_cue(m1, m2, keep=0.35):
    return np.maximum(partial_cue(X[m1], keep, 0.0), partial_cue(X[m2], keep, 0.0))


def noise_cue(n_on=4):
    y = np.zeros(X.shape[1])
    y[rng.choice(len(y), n_on, replace=False)] = 1
    return y


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def retrieve(
    x,
    recent=None,
    rho=0.0,
    steps=60,
    cue_gain=0.22,
    rec_gain=1.0,
    cross_inh=0.45,
    theta=0.50,
    alpha=0.35,
    adapt=0.06,
    conflict_tau=0.18,
    conflict_veto=0.75,
    # assembly-local commitment/disinhibition circuit
    commit_tau=0.24,
    commit_self=2.2,
    commit_bias=-4.8,
    completion_gain=2.0,
    cuebind_gain=1.5,
    commit_lateral=2.4,
    commit_conflict=5.0,
    assembly_conflict_gain=20.0,
    cue_conflict_gain=14.0,
    disinh_gain=0.8,
    apical_gain=5.0,
    coincidence_gain=4.5,
    apical_bias=-1.6,
    coincidence_bias=-3.8,
    branch_gain=6.0,
    branch_threshold=1.5,
    required_branches=1.5,
):
    d = W_cue @ x
    cue_support = np.tanh(d / 1.8)
    h = np.maximum(0.0, cue_gain * d)
    adaptation = np.zeros_like(h)
    recent_mask = np.zeros_like(h) if recent is None else H[recent]
    conflict_state = 0.0
    c = np.zeros(len(H))  # one local commitment cell/population per learned engram
    c_conflict = 0.0
    # Independent apical/context compartment: multiple branches must each see
    # a learned feature conjunction. It cannot be amplified by completion.
    branch_counts = np.einsum("mkd,d->mk", branches, x)
    branch_activity = sig(branch_gain * (branch_counts - branch_threshold))
    apical = sig(apical_gain * (branch_activity.sum(1) - required_branches))
    apical_support = np.maximum(0.0, (apical - 0.5) / 0.5)
    cue_pair_conflict = (
        apical_support.sum() ** 2 - np.dot(apical_support, apical_support)
    ) / 2.0
    cue_veto = 0.0
    c_trace = []

    for _ in range(steps):
        # Continuous inhibitory population; no ranked candidate or Boolean test.
        cue_veto_floor = sig(-3.0 * 0.35)
        cue_veto_target = max(
            0.0,
            (sig(3.0 * (cue_pair_conflict - 0.35)) - cue_veto_floor)
            / (1.0 - cue_veto_floor),
        )
        cue_veto = 0.70 * cue_veto + 0.30 * cue_veto_target
        recurrent = W_rec @ h
        incompatible = W_inh @ h
        raw_conflict = float(np.dot(h, incompatible) / (np.sum(h) + 1e-9))
        conflict_state = (
            1 - conflict_tau
        ) * conflict_state + conflict_tau * raw_conflict
        recurrent *= 1.0 + rho * recent_mask * cue_support

        # Local coincidence onto each engram's commitment population.
        # completion: fraction of that engram currently active.
        # cuebind: only activity that is also supported by the present cortical cue.
        completion = (H @ h) / H.sum(1)
        cuebind = (H @ (h * cue_support)) / H.sum(1)
        # A pooled interneuron can detect simultaneous completion of multiple
        # incompatible engrams without identifying/ranking either one.
        asm_pair_conflict = (
            completion.sum() ** 2 - np.dot(completion, completion)
        ) / 2.0

        # Simultaneous commitment activity creates a slow common veto.
        pair_conf = ((c.sum() ** 2 - np.dot(c, c)) / 2.0) / max(len(c) - 1, 1)
        c_conflict = 0.82 * c_conflict + 0.18 * pair_conf
        lateral = c.sum() - c
        # Basal recurrence and apical/context support must coincide.  The product
        # prevents a self-sustaining memory basin from committing without present input.
        basal = completion
        dendritic_coincidence = basal * apical
        c_drive = (
            commit_bias
            + coincidence_gain * dendritic_coincidence
            + completion_gain * (completion * apical_support)
            + cuebind_gain * cuebind
            + commit_self * c
            - commit_lateral * lateral
            - commit_conflict * c_conflict
            - assembly_conflict_gain * asm_pair_conflict
            - cue_conflict_gain * cue_veto
            - 2.0 * conflict_state
        )
        c_target = sig(c_drive)
        c = (1 - commit_tau) * c + commit_tau * c_target

        # Commitment is an actual state variable: it locally disinhibits only neurons
        # belonging to the assembly whose coincidence population is rising.
        local_gate = H.T @ c
        local_disinh = disinh_gain * local_gate * cue_support * recurrent
        u = (
            cue_gain * d
            + rec_gain * recurrent
            + local_disinh
            - cross_inh * incompatible
            - conflict_veto * conflict_state
            - theta
            - adaptation
        )
        target = sig(10 * u)
        target[target < 0.06] = 0.0
        h = (1 - alpha) * h + alpha * target
        adaptation = 0.9 * adaptation + 0.1 * adapt * h
        c_trace.append(c.copy())

    # Diagnostic decoding only; not used to select or gate retrieval.
    scores = H @ h / H.sum(1)
    order = np.argsort(scores)[::-1]
    best, second = order[:2]
    receptor_drive = W_return @ (h * (H.T @ c))
    return dict(
        h=h,
        best_memory=int(best),
        diagnostic_margin=float(scores[best] - scores[second]),
        commitment=float(c.max()),
        commitment_vector=c.copy(),
        commitment_trace=np.array(c_trace),
        conflict=float(conflict_state),
        commitment_conflict=float(c_conflict),
        assembly_conflict=float(asm_pair_conflict),
        cue_pair_conflict=float(cue_pair_conflict),
        cue_veto=float(cue_veto),
        apical_max=float(apical.max()),
        coincidence_max=float(dendritic_coincidence.max()),
        active_branches=int(np.sum(branch_activity > 0.5)),
        receptor_drive=receptor_drive,
    )


def benchmark(
    n=1000, keep=0.25, rho_wrong=0.0, committed_level=0.70, **retrieve_kwargs
):
    commits = correct = wrong_recent = wrong_receptor = 0
    commit_values = []
    for _ in range(n):
        m = int(rng.integers(len(X)))
        wrong = int((m + rng.integers(1, len(X))) % len(X))
        z = retrieve(
            partial_cue(X[m], keep), recent=wrong, rho=rho_wrong, **retrieve_kwargs
        )
        commit_values.append(z["commitment"])
        if z["commitment"] >= committed_level:
            commits += 1
            correct += z["best_memory"] == m
            wrong_recent += z["best_memory"] == wrong
            routed = int(np.argmax(z["receptor_drive"]))
            wrong_receptor += routed != receptors[z["best_memory"]]
    amb = []
    noise = []
    for _ in range(n):
        a, b = rng.choice(len(X), 2, replace=False)
        amb.append(
            retrieve(blended_cue(int(a), int(b)), **retrieve_kwargs)["commitment"]
        )
        noise.append(
            retrieve(noise_cue(int(rng.integers(2, 7))), **retrieve_kwargs)[
                "commitment"
            ]
        )
    amb = np.array(amb)
    noise = np.array(noise)
    commit_values = np.array(commit_values)
    return {
        "commit_rate": commits / n,
        "precision_when_committed": correct / max(commits, 1),
        "wrong_recent_win_rate": wrong_recent / n,
        "wrong_receptor_rate_given_commit": wrong_receptor / max(commits, 1),
        "median_commit_partial": float(np.median(commit_values)),
        "ambiguous_high_commit_rate": float(np.mean(amb >= committed_level)),
        "median_commit_ambiguous": float(np.median(amb)),
        "noise_high_commit_rate": float(np.mean(noise >= committed_level)),
        "median_commit_noise": float(np.median(noise)),
    }


if __name__ == "__main__":
    for keep in (0.15, 0.25, 0.35, 0.50):
        print("keep", keep, benchmark(n=800, keep=keep, rho_wrong=0.0))
    for rho in (0.0, 0.5, 1.0, 2.0, 4.0):
        print("wrong recency", rho, benchmark(n=800, keep=0.25, rho_wrong=rho))
