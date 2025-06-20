import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv("rcnn_loss_log.csv")

# Plot all losses over steps
plt.figure(figsize=(8, 6))

plt.plot(df["step"], df["total_loss"], label="Total Loss", linewidth=2)
plt.plot(df["step"], df["loss_classifier"], label="Loss Classifier", linestyle="--")
plt.plot(df["step"], df["loss_box_reg"], label="Loss Box Reg", linestyle="--")
plt.plot(df["step"], df["loss_objectness"], label="Loss Objectness", linestyle="--")
plt.plot(df["step"], df["loss_rpn_box_reg"], label="Loss RPN Box Reg", linestyle="--")

# Styling
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("RCNN Losses over Training Steps")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()