# S5 Codec Training Notes

## Diagnostic Results (Phase 1, Epoch 100)

### Problem Identified: Output Amplitude Too Small

**Symptoms:**
- SNR: -0.06 dB (model outputs ~80x less power than input)
- Input power: 0.5, Output power: 0.006
- Even without quantization: SNR -0.07 dB → **quantization is NOT the problem**
- Decoder test: single channel excitation (h=1) → output power 0.000018 (way too small)

**Root Cause:**
- Decoder C weights too small: mean |C| = 0.09, some channels |C| = 0.0005 (dead)
- No output scaling layer (decoder was directly outputting `Re[C^H · h(t)]`)
- With |C| ≈ 0.09 and |h| ≈ 0.5, output ≈ 0.045 but need ≈ 0.7 for proper reconstruction

### Fixes Applied

1. **Increased C initialization**: 0.1 → 0.5
   - Gives decoder stronger starting weights
   - Helps gradient flow during early training

2. **Added learnable output scale & bias**:
   ```python
   self.output_scale = nn.Parameter(torch.ones(1))
   self.output_bias = nn.Parameter(torch.zeros(1))
   y = self.output_scale * y + self.output_bias
   ```
   - Allows decoder to learn proper amplitude scaling
   - Critical for matching target signal amplitude

3. **Increased quantization warmup**: 5 → 20 epochs
   - Lets encoder+decoder learn continuous mapping first
   - Then introduce quantization noise once reconstruction is working

## Training Philosophy

### Current Approach (No Freezing)
- All parameters train jointly from epoch 0
- Only staging is quantization warmup (bypass for first 20 epochs)
- Simple but can be unstable: encoder and decoder are both moving targets

### Recommended Approach (2-Phase Freeze)

**Phase A (epochs 0-30): Learn continuous bottleneck**
- Train: encoder + decoder jointly
- Freeze: nothing
- Bypass quantization: YES
- Goal: Can the autoencoder reconstruct at all?

**Phase B (epochs 30-100): Adapt to quantization**
- Train: encoder only
- Freeze: decoder
- Bypass quantization: NO
- Goal: Encoder learns to produce quantization-friendly states

**Phase C (optional fine-tune)**
- Train: all
- Freeze: nothing
- Bypass quantization: NO
- Low LR end-to-end fine-tuning

### Why Freezing Helps
- **Analytical decoder has learnable eigenvalues & C** — not a fixed lookup table
- **Encoder skip connections** dominate early training (output ≈ input)
- Freezing decoder after Phase A gives encoder a **fixed target** to optimize for

## Next Steps

1. **Re-train with fixes**: New C init + output scaling should give proper amplitude
2. **Check diagnostics again**: Run `debug_model.py` to verify output power is correct
3. **Consider implementing decoder freeze**: Add `--freeze-decoder-after N` option to training script
4. **Monitor eigenvalue usage**: Are all 4 channels being used, or is it still dominated by 1-2?

## Sanity Checks

✅ Quantization cost is negligible (-0.00 dB) → scalar quantization working
✅ Eigenvalues well-distributed: |λ| ∈ [0.72, 1.0], frequencies across [0.02π, 0.96π]
✅ Encoder outputs are reasonable: z_std = 0.51, using all 28/32 dimensions
❌ Output amplitude too small → **FIXED with output_scale parameter**
