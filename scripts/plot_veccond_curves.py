"""Plot arm-minus-native curves for all veccond runs, long-interruption context."""
import glob, json, sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit('matplotlib not available')

REPO = '/home/user/sonic-manifold-flow'
CONTEXT = 'long_interruption_crossover'
UPDATES_PER_PASS = 13570

RUNS = [
    dict(label='Linear 64d',          color='#2196F3', linestyle='-',
         kind='v1', eval_dir=f'{REPO}/runs/shared_goal/veccond_240pass_evals',
         fallback=f'{REPO}/runs/shared_goal/veccond_80pass_evals'),
    dict(label='Dendritic 64d',       color='#E91E63', linestyle='-',
         kind='v2', eval_dir=f'{REPO}/runs/shared_goal/veccond2_240pass_evals',
         fallback=f'{REPO}/runs/shared_goal/veccond2_80pass_evals'),
    dict(label='Dendritic-K8 64d',    color='#4CAF50', linestyle='-',
         kind='v2', eval_dir=f'{REPO}/runs/shared_goal/veccond3_240pass_evals',
         fallback=None),
    dict(label='Linear 128d',         color='#2196F3', linestyle='--',
         kind='v1', eval_dir=f'{REPO}/runs/shared_goal/veccond_240pass_2x_evals',
         fallback=f'{REPO}/runs/shared_goal/veccond_80pass_2x_evals'),
    dict(label='Dendritic 128d',      color='#E91E63', linestyle='--',
         kind='v2', eval_dir=f'{REPO}/runs/shared_goal/veccond2_240pass_2x_evals',
         fallback=f'{REPO}/runs/shared_goal/veccond2_80pass_2x_evals'),
    dict(label='Dendritic-K8 128d',   color='#4CAF50', linestyle='--',
         kind='v2', eval_dir=f'{REPO}/runs/shared_goal/veccond3_240pass_2x_evals',
         fallback=None),
]

ARMS = ['A_k1_nooledger', 'B_k1_ledger', 'C_k4_nooledger', 'D_k4_ledger']
ARM_LABELS = {'A_k1_nooledger': 'A', 'B_k1_ledger': 'B',
              'C_k4_nooledger': 'C', 'D_k4_ledger': 'D'}


def load_v2(eval_dir):
    rows = {}
    for f in sorted(glob.glob(f'{eval_dir}/eval_mark_*.json')):
        with open(f) as fh:
            d = json.load(fh)
        passes = d['snapshot']['updates_done'] / UPDATES_PER_PASS
        ctx = d['evaluation']['context_modes'][CONTEXT]
        for arm in ARMS:
            c = ctx['arm_summaries'][arm]['all']['sampled']['contrasts']['arm_minus_native']
            rows.setdefault(arm, []).append((passes, c['mean'], c['ci95_low'], c['ci95_high']))
    return {arm: sorted(v) for arm, v in rows.items()}


def load_v1(eval_dir):
    rows = {}
    for f in sorted(glob.glob(f'{eval_dir}/eval_*.json')):
        with open(f) as fh:
            d = json.load(fh)
        passes = d.get('snapshot', {}).get('passes', float('nan'))
        ctx = d.get('evaluation', {}).get('context_modes', {}).get(CONTEXT, {})
        for arm in ARMS:
            arm_data = ctx.get('arm_summaries', {}).get(arm, {})
            c = arm_data.get('all', {}).get('sampled', {}).get('contrasts', {}).get('arm_minus_native')
            if c:
                rows.setdefault(arm, []).append((passes, c['mean'], c['ci95_low'], c['ci95_high']))
    return {arm: sorted(v) for arm, v in rows.items()}


def load_run(run):
    loader = load_v1 if run['kind'] == 'v1' else load_v2
    data = loader(run['eval_dir'])
    if not any(data.values()):
        fb = run.get('fallback')
        if fb:
            data = loader(fb)
    return data


fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True, sharex=True)
fig.suptitle('arm-minus-native  |  long-interruption  |  sampled\n(* = 95% CI above zero)', fontsize=13)

for ax, arm in zip(axes.flat, ARMS):
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axhline(0.0408, color='grey', linewidth=0.8, linestyle=':',
               label='v1 baseline @pass8 (+0.041)')
    for run in RUNS:
        data = load_run(run)
        pts = data.get(arm, [])
        if not pts:
            continue
        xs  = [p[0] for p in pts]
        ys  = [p[1] for p in pts]
        los = [p[2] for p in pts]
        his = [p[3] for p in pts]
        ax.plot(xs, ys, marker='o', markersize=3,
                color=run['color'], linestyle=run['linestyle'], label=run['label'])
        ax.fill_between(xs, los, his, alpha=0.10, color=run['color'])
        for x, y, lo in zip(xs, ys, los):
            if lo > 0:
                ax.plot(x, y, '*', color=run['color'], markersize=8)
    ax.set_title(f'Arm {ARM_LABELS[arm]}', fontsize=11)
    ax.set_xlabel('training passes')
    ax.set_ylabel('mean margin delta')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper left')

plt.tight_layout()
out = f'{REPO}/runs/shared_goal/veccond_curves_all.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'saved: {out}')
