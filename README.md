# Sparse autoencoders

This repository hosts sparse autoencoders trained on the GPT2-small model's activations.

### Install

```sh
pip install git+https://github.com/openai/sparse_autoencoder.git
```

### Code structure

See [model.py](./sparse_autoencoder/model.py) for details on the autoencoder model architecture.
See [paths.py](./sparse_autoencoder/paths.py) for more details on the available autoencoders.

### Example usage

```py
import torch
import blobfile as bf
import transformer_lens
import sparse_autoencoder

# Load the autoencoder
layer_index = 0
location = "resid_post_mlp"
layernorm = True
with bf.BlobFile(sparse_autoencoder.paths.v5_32k(location, layer_index), mode="rb") as f:
    state_dict = torch.load(f)
    autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)

# Extract neuron activations with transformer_lens
model = transformer_lens.HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)
prompt = "This is an example of a prompt that"
tokens = model.to_tokens(prompt)  # (1, n_tokens)
with torch.no_grad():
    logits, activation_cache = model.run_with_cache(tokens, remove_batch_dim=True)

transformer_lens_loc_map = {
    "mlp_post_act": f"blocks.{layer_index}.mlp.hook_post",
    "resid_delta_attn": f"blocks.{layer_index}.hook_attn_out",
    "resid_post_attn": f"blocks.{layer_index}.hook_resid_mid",
    "resid_delta_mlp": f"blocks.{layer_index}.hook_mlp_out",
    "resid_post_mlp": f"blocks.{layer_index}.hook_resid_post",
}
input_tensor = activation_cache[transformer_lens_loc_map[location]]

# Encode neuron activations with the autoencoder
device = next(model.parameters()).device
autoencoder.to(device)

input_tensor_ln = input_tensor
if layernorm:
    # apply layer norm first
    mu = input_tensor_ln.mean(dim=1, keepdim=True)
    input_tensor_ln = input_tensor_ln - mu
    std = input_tensor_ln.std(dim=1, keepdim=True)
    input_tensor_ln = input_tensor_ln / std

with torch.no_grad():
    latent_activations = autoencoder.encode(input_tensor_ln)  # (n_tokens, n_latents)
    reconstructed_activations = autoencoder.decode(latent_activations)

if layernorm:
    reconstructed_activations = reconstructed_activations * std + mu

normalized_mse = (reconstructed_activations - input_tensor).pow(2).sum(dim=1) / (input_tensor).pow(2).sum(dim=1)
```
