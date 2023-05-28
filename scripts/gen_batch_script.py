def gen_for_ds(ds, seed, fold_idx):
    return [
        f"python3 main.py data={ds} evaluation=finetune_ensemble +evaluation.seed={seed} +evaluation.fold_idx={fold_idx}"
    ]

lines = []
seeds = [ 46774, 5111, 24539, 11129, 54277, 33478 ]

hyper_settings = [
    [ 1, 1 ],
    [ 2, 2 ],
    [ 3, 3 ],
    [ 2, 1 ],
    [ 3, 1 ],
    [ 3, 2 ],
    [ 1, 2 ],
    [ 1, 3 ],
]

NUM_FOLDS = {
    # 'wisdm_10s': 5,
    'adl_10s': 7,
    # 'oppo_10s': 4,
    # 'realworld_10s': 15,
    # 'pamap_10s': 8
}

for ds in NUM_FOLDS.keys():
    for seed in seeds:
        for fold_idx in range(NUM_FOLDS[ds]):
            # lines.append(f"./scripts/eval.sh {ds} finetune {seed} {fold_idx}")
            for hyper in hyper_settings:
                lines.extend(gen_for_ds(ds, seed, fold_idx))

# shuffle lines
import random
random.shuffle(lines)

with open('run_classification_all.sh', 'w') as f:
    f.write('\n'.join(lines))