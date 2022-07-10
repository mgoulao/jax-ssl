import os
import jax
import numpy as np

def save_model(args, params, state, n_checkpoint):
  checkpoint_file_path = os.path.join(args.checkpoints_dir, f"checkpoint_{n_checkpoint}.npy")
  print(f"Saving checkpoint to: {checkpoint_file_path}")
  to_device_fn = lambda x: np.array(jax.device_get(x))
  np_params = jax.tree_map(to_device_fn, params)
  np_state = jax.tree_map(to_device_fn, state)
  with open(checkpoint_file_path, "wb") as fp:
    np.save(fp, (np_params, np_state))

def load_model(args):
  params, state = np.load(args.checkpoint_file, allow_pickle=True)
  return params, state