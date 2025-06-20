import pandas as pd
import matplotlib.pyplot as plt

# Weights
w_mask = 0.3
w_ret = 0.4
lambda_neg = 0.15

# Read CSV
df = pd.read_csv("object_location_loss_log.csv")

# Compute weighted components
df["weighted_loss_mask"] = w_mask * df["loss_mask"] / (w_mask + w_ret + 2 * lambda_neg)
df["weighted_loss_retina"] = w_ret * df["loss_retina"] / (w_mask + w_ret + 2 * lambda_neg)
df["weighted_fp_loss_retina"] = lambda_neg * df["fp_loss_retina"] / (w_mask + w_ret + 2 * lambda_neg)
df["weighted_fp_loss_maskrcnn"] = lambda_neg * df["fp_loss_maskrcnn"] / (w_mask + w_ret + 2 * lambda_neg)

# Total loss
df["total_loss"] = (
    df["weighted_loss_mask"] +
    df["weighted_loss_retina"] +
    df["weighted_fp_loss_retina"] +
    df["weighted_fp_loss_maskrcnn"]
)

# Plot
plt.figure(figsize=(8, 6))

plt.plot(df["step"], df["total_loss"], label="Total Loss", linewidth=2)
plt.plot(df["step"], df["weighted_loss_mask"], label="Weighted Loss Mask", linestyle="--")
plt.plot(df["step"], df["weighted_loss_retina"], label="Weighted Loss Retina", linestyle="--")
# plt.plot(df["step"], df["weighted_fp_loss_retina"], label="Weighted FP Loss Retina", linestyle=":")
# plt.plot(df["step"], df["weighted_fp_loss_maskrcnn"], label="Weighted FP Loss MaskRCNN", linestyle=":")

# Styling
plt.xlabel("Step")
plt.ylabel("Weighted Loss Contribution")
plt.title("Weighted Loss Components and Total Loss over Steps")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()