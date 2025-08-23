import torch
import torch.nn as nn
import torchlens as tl  # ★ import as a module; tl.show_model_graph used below

class MyTransformer(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=2, num_heads=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.decoder(x)

# Optional wrapper not required for TorchLens; it will trace submodules anyway.
transformer = MyTransformer()

# Example input (B, T, D_in)
x = torch.randn(2, 5, 128)

# Option A: one-liner that logs activations and ALSO renders the graph.
log = tl.log_forward_pass(transformer, x, layers_to_save=None, vis_opt="unrolled")


# --- Make a wrapper that exposes internals clearly ---
class TransformerWrapper(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.model = transformer

    def forward(self, x):
        # Instead of doing everything in one call, we explicitly call submodules
        x = self.model.embedding(x)
        # Go through each encoder layer explicitly so TorchLens can track them
        for i, layer in enumerate(self.model.encoder.layers):
            x = layer(x)
        x = self.model.decoder(x)
        return x

# --- Instantiate and wrap the model ---
transformer = MyTransformer()
wrapped_model = TransformerWrapper(transformer)

# --- Dummy input ---
x = torch.randn(2, 5, 128)

# --- Run TorchLens logging ---
log = log_forward_pass(wrapped_model, x)

# --- Visualize graph ---
show_model_graph(log)
