import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
invalidator_tp = []
invalidator_fp = []
invalidator_f1 = []
count = 0
with open("result.txt", "r") as f:
    for line in f:
        count +=1
        if count > 101:
            continue
        tp, fp, f1 = line.split("\t")
        invalidator_tp.append(int(tp))
        invalidator_fp.append(int(fp))
        invalidator_f1.append(float(f1))

threshold = [i*0.005 for i in range(len(invalidator_tp))]

fig = plt.figure()
ax = fig.add_subplot(111)

lns1 = ax.plot(threshold, invalidator_f1, color = "tab:orange", label = 'F1-score')
ax2 = ax.twinx()
lns2 = ax2.plot(threshold, invalidator_tp, color = "tab:green", label = 'True Positive')
lns3 = ax2.plot(threshold, invalidator_fp, color = "tab:blue", label = 'False Positive')
ax2.fill_between(threshold, invalidator_tp, invalidator_fp, color='tab:green', alpha=0.3)
plt.vlines(0,0,163, color="tab:red", linestyles='dashed')
plt.vlines(0.23,0,163, color="tab:red", linestyles='dashed')
# # added these three lines
lns = lns1+lns2+lns3
print(lns)
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

ax.grid()
ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax2.set_ylabel("Cases")
ax2.set_ylim(0, 163)
ax.set_ylim(0,1)
plt.savefig('threshold.png', dpi=300)

