import os
import pickle
import random
import argparse
import numpy as np


def load_dataset(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def count_transitions(episodes):
    return sum(len(ep["actions"]) for ep in episodes)


def filter_complement_transitions(episodes, criterion):
    """Split episodes into sub-episodes keeping only transitions in the complement region.

    criterion choices:
      'action_positive'       — keep transitions where NOT(ax>=0 AND ay>=0)
      'state_positive'        — keep transitions where NOT(x>=0 AND y>=0)
      'state_action_positive' — keep transitions where NOT(x>=0 AND y>=0 AND ax>=0 AND ay>=0)
    """
    filtered = []
    for ep in episodes:
        obs  = ep['observations']   # (T+1, state_dim)
        acts = ep['actions']        # (T, action_dim)
        T    = len(acts)

        if criterion == 'action_positive':
            mask = (acts[:, 0] < 0) | (acts[:, 1] < 0)
        elif criterion == 'state_positive':
            states = obs[:-1]
            mask = (states[:, 0] < 0) | (states[:, 1] < 0)
        elif criterion == 'state_action_positive':
            states = obs[:-1]
            mask = ((states[:, 0] < 0) | (states[:, 1] < 0)) & \
                   ((acts[:, 0] < 0) | (acts[:, 1] < 0))
        else:
            raise ValueError(f"Unknown complement criterion: {criterion!r}")

        # Build contiguous sub-episodes of masked transitions
        i = 0
        while i < T:
            if mask[i]:
                j = i
                while j < T and mask[j]:
                    j += 1
                infos = ep['infos'][i:j] if isinstance(ep.get('infos'), list) else []
                filtered.append({
                    'observations': obs[i:j + 1],
                    'actions':      acts[i:j],
                    'rewards':      ep['rewards'][i:j],
                    'terminals':    ep['terminals'][i:j],
                    'truncations':  ep['truncations'][i:j],
                    'infos':        infos,
                })
                i = j
            else:
                i += 1
    return filtered


def mix_datasets_by_transitions(
    dataset_sym,
    dataset_asym,
    alpha,
    target_transitions,
    seed=0,
    complement_filter=None,
    sym_filter=None,
):
    random.seed(seed)

    sym_eps = dataset_sym["episodes"]
    asym_eps = dataset_asym["episodes"]

    if sym_filter is not None:
        sym_eps = filter_complement_transitions(sym_eps, sym_filter)
        print(f"Sym complement filter '{sym_filter}': {len(sym_eps)} sub-episodes, "
              f"{count_transitions(sym_eps)} transitions retained")

    if complement_filter is not None:
        asym_eps = filter_complement_transitions(asym_eps, complement_filter)
        print(f"Asym complement filter '{complement_filter}': {len(asym_eps)} sub-episodes, "
              f"{count_transitions(asym_eps)} transitions retained")

    random.shuffle(sym_eps)
    random.shuffle(asym_eps)

    mixed_eps = []

    total = 0
    sym_idx = 0
    asym_idx = 0

    while total < target_transitions:
        use_sym = random.random() < alpha

        if use_sym and sym_idx < len(sym_eps):
            ep = sym_eps[sym_idx]
            sym_idx += 1
        elif asym_idx < len(asym_eps):
            ep = asym_eps[asym_idx]
            asym_idx += 1
        else:
            break

        mixed_eps.append(ep)
        total += len(ep["actions"])

    print(f"Target: {target_transitions}")
    print(f"Achieved: {total}")
    print(f"Episodes used: {len(mixed_eps)}")

    return mixed_eps


def save_dataset(episodes, metadata, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    dataset = {
        "episodes": episodes,
        "metadata": metadata,
    }

    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym_path",  type=str,
                        default="./datasets/particle_circle_2d_random_uniform.pkl")
    parser.add_argument("--asym_path", type=str,
                        default="./datasets/particle_circle_2d_action_positive_flat.pkl")
    parser.add_argument("--alpha",               type=float, required=True)
    parser.add_argument("--target_transitions",  type=int,   default=30_000)
    parser.add_argument("--seed",                type=int,   default=0)
    parser.add_argument("--output_path",         type=str,   required=True)
    parser.add_argument(
        "--sym_filter",
        type=str, default=None,
        choices=["action_positive", "state_positive", "state_action_positive"],
        help="Filter sym dataset to the complement of this bias quadrant before mixing.",
    )
    parser.add_argument(
        "--complement_filter",
        type=str, default=None,
        choices=["action_positive", "state_positive", "state_action_positive"],
        help="Filter asym dataset to complement of the given bias quadrant before mixing.",
    )

    args = parser.parse_args()

    dataset_sym  = load_dataset(args.sym_path)
    dataset_asym = load_dataset(args.asym_path)

    mixed_eps = mix_datasets_by_transitions(
        dataset_sym,
        dataset_asym,
        alpha=args.alpha,
        target_transitions=args.target_transitions,
        seed=args.seed,
        complement_filter=args.complement_filter,
        sym_filter=args.sym_filter,
    )

    metadata = {
        "alpha":              args.alpha,
        "target_transitions": args.target_transitions,
        "actual_transitions": sum(len(ep["actions"]) for ep in mixed_eps),
        "source_sym":         args.sym_path,
        "source_asym":        args.asym_path,
        "seed":               args.seed,
        "sym_filter":         args.sym_filter,
        "complement_filter":  args.complement_filter,
    }

    save_dataset(mixed_eps, metadata, args.output_path)


if __name__ == "__main__":
    main()