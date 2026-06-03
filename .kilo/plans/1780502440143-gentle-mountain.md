# Plan: Implement Train Tokenizer from Scratch

The objective is to create a system to train the `KronosTokenizer` from scratch, mirroring the existing finetuning pipeline.

## Analysis
- The existing `finetune_tokenizer.py` provides a complete DDP training loop.
- The data feeding, including Z-score normalization per window and clipping, is implemented in `QlibDataset`.
- The `KronosTokenizer` architecture parameters were extracted from the pretrained `NeoQuasar/Kronos-Tokenizer-base` model.

## Implementation Steps

### 1. Create `train_tokenizer.py`
- Duplicate the logic from `finetune_tokenizer.py` to a new file `train_tokenizer.py`.
- Replace the pretrained model loading logic:
    - Remove `KronosTokenizer.from_pretrained(config['pretrained_tokenizer_path'])`.
    - Initialize `KronosTokenizer` directly with the following parameters:
        - `d_in`: 22 (4 OHLC + 18 features from features.py)
        - `d_model`: 256
        - `n_heads`: 4
        - `ff_dim`: 512
        - `n_enc_layers`: 4
        - `n_dec_layers`: 4
        - `ffn_dropout_p`: 0.0
        - `attn_dropout_p`: 0.0
        - `resid_dropout_p`: 0.0
        - `s1_bits`: 10
        - `s2_bits`: 10
        - `beta`: 0.05
        - `gamma0`: 1.0
        - `gamma`: 1.1
        - `zeta`: 0.05
        - `group_size`: 4
- Keep the DDP setup, `QlibDataset` usage, and training loop identical to `finetune_tokenizer.py` to ensure consistency in data feeding and normalization.
- Override `config.tokenizer_save_folder_name` to `'train_tokenizer_demo'` to avoid overwriting finetuning results.

### 2. Verification
- The script should be launchable via `torchrun` similarly to the finetuning script:
  `torchrun --standalone --nproc_per_node=NUM_GPUS train_tokenizer.py`
- Verify that the model is initialized from scratch and not loading any pretrained weights.
- Ensure that the `QlibDataset` is correctly providing normalized windows.

## Considerations
- **Hyperparameters**: Use the same learning rate and optimizer settings as in `finetune_tokenizer.py` unless specified otherwise, as it's requested to be "on the same lines".
- **Normalization**: Since `QlibDataset` is reused, Z-score normalization per window is guaranteed.
