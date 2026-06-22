import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("log.csv")
df.columns = df.columns.str.strip()

plt.figure(figsize=(12,6))



#   logfile << "t, vel_des, vel_act, w_des, w_act, zero, roll, pitch, height_target, height\n";




# plt.plot(df["t"], df["vel_des"], linewidth=2, label="vel_cmd")
# plt.plot(df["t"], df["vel_act"], linewidth=2, label="vel_act")
# plt.plot(df["t"], df["w_des"], linewidth=2, label="w_cmd")
# plt.plot(df["t"], df["w_act"], linewidth=2, label="w_act")
plt.plot(df["t"], df["zero"], linewidth=2, label="zero", color="red", linestyle="--")
plt.plot(df["t"], df["roll"], linewidth=2, label="roll", color="blue")
plt.plot(df["t"], df["pitch"], linewidth=2, label="pitch", color="orange")
# plt.plot(df["t"], df["height_target"], linewidth=2, label="height_target")
# plt.plot(df["t"], df["height"], linewidth=2, label="height")

plt.xlabel("Time (s)", fontsize=20)
plt.ylabel("velocity (m/s)", fontsize=20)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.xlim(0, 20)
plt.ylim(-0.1, 0.1)

plt.legend(loc="upper right", frameon=False, fontsize=20)
plt.grid(False)

plt.tight_layout()

plt.savefig(
    "vel.pdf",
    bbox_inches="tight"
)

plt.show()
