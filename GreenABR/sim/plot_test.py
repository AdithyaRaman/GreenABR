import numpy as np
import matplotlib.pyplot as plt


LOG_PATH = 'results/hd_reward/log_test'
#LOG_PATH = 'results/hd_reward_log/log_test'
PLOT_SAMPLES = 300


epoch = []
rewards_min = []
rewards_5per = []
rewards_mean = []
rewards_median = []
rewards_95per = []
rewards_max = []

with open(LOG_PATH, 'rb') as f:
    for line in f:
        parse = line.split()
        epoch.append(float(parse[0]))
        rewards_min.append(float(parse[1]))
        rewards_5per.append(float(parse[2]))
        rewards_mean.append(float(parse[3]))
        rewards_median.append(float(parse[4]))
        rewards_95per.append(float(parse[5]))
        rewards_max.append(float(parse[6]))

f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True)

ax1.plot(epoch[-PLOT_SAMPLES:], rewards_min[-PLOT_SAMPLES:])
ax1.set_ylabel('rewards_min')

ax2.plot(epoch[-PLOT_SAMPLES:], rewards_5per[-PLOT_SAMPLES:])
ax2.set_ylabel('rewards_5per')

ax3.plot(epoch[-PLOT_SAMPLES:], rewards_mean[-PLOT_SAMPLES:])
ax3.set_ylabel('rewards_mean')

ax4.plot(epoch[-PLOT_SAMPLES:], rewards_median[-PLOT_SAMPLES:])
ax4.set_ylabel('rewards_median')

ax5.plot(epoch[-PLOT_SAMPLES:], rewards_95per[-PLOT_SAMPLES:])
ax5.set_ylabel('rewards_95per')

ax6.plot(epoch[-PLOT_SAMPLES:], rewards_max[-PLOT_SAMPLES:])
ax6.set_ylabel('rewards_max')
ax6.set_xlabel('epoch')

f.subplots_adjust(hspace=0)

#plt.savefig('reward_plot.png')
plt.show()