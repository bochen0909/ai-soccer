"""Adds the soft-pass acts to policies trained before they existed: one act per teammate slot, after the lofts.

The new rows start as copies of the firm-pass rows with a lower bias, so a converted policy keeps playing the firm
passes it learned and can learn when a soft one is better. Rewrites the files in place; files that already have the
acts are left alone. Works on exported JSON policies and on PyTorch checkpoints.

  python3 python/add_soft_passes.py models/trained-5v5.json runs/role_gen15/latest.pt ...
"""
import base64
import json
import sys

import numpy as np

ACT_PASS_BASE = 6
SOFT_BIAS_OFFSET = -2.0


def roster_size(obs_size):
    """Observation rows are 63 + 21n (ego, ball, match, teammates, opponents) + 23 + n (role block)."""
    slots, remainder = divmod(obs_size - 86, 22)
    if remainder or slots < 1:
        raise ValueError(f"observation size {obs_size} matches no roster size")
    return slots


def extend(weight, bias, slots):
    """weight [acts, hidden], bias [acts]: none, three shots, tackle, save, then firm pass and loft per slot."""
    acts = weight.shape[0]
    if acts == ACT_PASS_BASE + 3 * slots:
        return None
    if acts != ACT_PASS_BASE + 2 * slots:
        raise ValueError(f"{acts} acts do not fit {slots} slots")
    firm = slice(ACT_PASS_BASE, ACT_PASS_BASE + slots)
    return (np.concatenate([weight, weight[firm]], axis=0),
            np.concatenate([bias, bias[firm] + SOFT_BIAS_OFFSET], axis=0))


def decode(tensor):
    return np.frombuffer(base64.b64decode(tensor["data"]), dtype="<f4").reshape(tensor["shape"])


def encode(array):
    array = np.ascontiguousarray(array, dtype="<f4")
    return {"shape": list(array.shape), "data": base64.b64encode(array.tobytes()).decode("ascii")}


def convert_json(path):
    with open(path) as handle:
        document = json.load(handle)
    slots = roster_size(document["obsSize"])
    networks = [document["tensors"]] + ([document["keeper"]["tensors"]] if "keeper" in document else [])
    changed = False
    for tensors in networks:
        extended = extend(decode(tensors["actW"]), decode(tensors["actB"]), slots)
        if extended is None:
            continue
        tensors["actW"], tensors["actB"] = encode(extended[0]), encode(extended[1])
        changed = True
    if changed:
        document["actCount"] = ACT_PASS_BASE + 3 * slots
        with open(path, "w") as handle:
            json.dump(document, handle, separators=(",", ":"))
    return changed


def convert_checkpoint(path):
    import torch
    checkpoint = torch.load(path)
    state = checkpoint["policy"]
    slots = roster_size(state["body.0.weight"].shape[1])
    changed = False
    for prefix in ["", "keeper_net."]:
        if prefix + "act.weight" not in state:
            continue
        extended = extend(state[prefix + "act.weight"].numpy(), state[prefix + "act.bias"].numpy(), slots)
        if extended is None:
            continue
        state[prefix + "act.weight"], state[prefix + "act.bias"] = torch.from_numpy(extended[0]), torch.from_numpy(extended[1])
        changed = True
    if changed:
        torch.save(checkpoint, path)
    return changed


if __name__ == "__main__":
    for path in sys.argv[1:]:
        changed = convert_json(path) if path.endswith(".json") else convert_checkpoint(path)
        print(("converted " if changed else "already has soft passes: ") + path)
