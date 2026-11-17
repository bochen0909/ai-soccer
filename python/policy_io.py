"""Writes a policy's weights as the JSON file the game loads (exact float32 values, base64)."""
import base64
import json
import os

TENSORS = {  # name in the file -> name in the PyTorch state dict
    "w1": "body.0.weight", "b1": "body.0.bias", "w2": "body.2.weight", "b2": "body.2.bias",
    "moveW": "move.weight", "moveB": "move.bias", "sprintW": "sprint.weight", "sprintB": "sprint.bias",
    "actW": "act.weight", "actB": "act.bias",
}


def _tensors(state, prefix=""):
    tensors = {}
    for name, key in TENSORS.items():
        array = state[prefix + key].detach().cpu().numpy().astype("<f4")
        tensors[name] = {"shape": list(array.shape), "data": base64.b64encode(array.tobytes()).decode("ascii")}
    return tensors


def write_policy_json(state, path, source="", iteration=None):
    """state is a PyTorch state dict of train_ppo.Policy or TeamPolicy. Returns the file size in bytes.

    A TeamPolicy also writes a "keeper" block with the goalkeeper network's tensors; files without it use one network for everyone.
    """
    tensors = _tensors(state)
    document = {"version": 1, "obsSize": state["body.0.weight"].shape[1], "hidden": state["body.0.weight"].shape[0],
                "actCount": state["act.weight"].shape[0], "moveCount": state["move.weight"].shape[0],
                "source": source, "iteration": iteration, "tensors": tensors}
    if "keeper_net.body.0.weight" in state:
        document["keeper"] = {"tensors": _tensors(state, "keeper_net.")}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(document, handle, separators=(",", ":"))
    return os.path.getsize(path)
