import os
import numpy as np
import matplotlib.pyplot as plt


RESULTS_FOLDER = 'test_results/'
NUM_BINS = 100
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 44
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
K_IN_M = 1000.0
REBUF_P = 4.3
SMOOTH_P = 1
COLOR_MAP = plt.cm.jet #nipy_spectral, Set1,Paired 
SIM_DP = 'sim_dp'
SCHEMES = ['hd_reward', 'phone_vmaf', 'regular_vmaf']
#SCHEMES = ['hd_reward']
PLOT_FOLDER = './plots/'


font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }



def main():
	video_chunk_all={}
	time_all = {}
	bit_rate_all = {}
	buff_all = {}
	bw_all = {}
	raw_reward_all = {}
	pVMAF_all = {}
	rMVAF_all = {}


	for scheme in SCHEMES:
		video_chunk_all[scheme]={}
		time_all[scheme] = {}
		raw_reward_all[scheme] = {}
		bit_rate_all[scheme] = {}
		buff_all[scheme] = {}
		bw_all[scheme] = {}
		pVMAF_all[scheme]={}
		rMVAF_all[scheme]={}

	log_files = os.listdir(RESULTS_FOLDER)
	for log_file in log_files:
		video_chunk=[]
		time_ms = []
		bit_rate = []
		buff = []
		bw = []
		reward = []
		pVMAF=[]
		rVMAF=[]

		print log_file

		with open(RESULTS_FOLDER + log_file, 'rb') as f:
			if SIM_DP in log_file:
				last_t = 0
				last_b = 0
				last_q = 1
				lines = []
				for line in f:
					lines.append(line)
					parse = line.split()
					if len(parse) >= 6:
						video_chunk.append(float(parse[0]))
						time_ms.append(float(parse[4]))
						bit_rate.append(VIDEO_BIT_RATE[int(parse[7])])
						buff.append(float(parse[5]))
						bw.append(float(parse[6]))
				
				for line in reversed(lines):
					parse = line.split()
					r = 0
					if len(parse) > 1:
						t = float(parse[4])
						b = float(parse[5])
						q = int(parse[7])
						if b == 4:
							rebuff = (t - last_t) - last_b
							assert rebuff >= -1e-4
							r -= REBUF_P * rebuff

						r += VIDEO_BIT_RATE[q] / K_IN_M
						r -= SMOOTH_P * np.abs(VIDEO_BIT_RATE[q] - VIDEO_BIT_RATE[last_q]) / K_IN_M
						reward.append(r)

						last_t = t
						last_b = b
						last_q = q

			else:
				for line in f:
					parse = line.split()
					if len(parse) <= 1:
						break
					time_ms.append(float(parse[1]))
					bit_rate.append(int(parse[2]))
					buff.append(float(parse[3]))
					bw.append(float(parse[5]) / float(parse[6]) * BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
					reward.append(float(parse[7]))
					pVMAF.append(float(parse[8]))
					rVMAF.append(float(parse[9]))

		if SIM_DP in log_file:
			time_ms = time_ms[::-1]
			bit_rate = bit_rate[::-1]
			buff = buff[::-1]
			bw = bw[::-1]
		
		time_ms = np.array(time_ms)
		time_ms -= time_ms[0]
		
		# print log_file

		for scheme in SCHEMES:
			if scheme in log_file:
				time_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = time_ms
				bit_rate_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bit_rate
				buff_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = buff
				bw_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw
				raw_reward_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = reward
				pVMAF_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = pVMAF
				rMVAF_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = rVMAF
				break

	# ---- ---- ---- ----
	# Reward records
	# ---- ---- ---- ----
		
	log_file_all = []
	reward_all = {}
	for scheme in SCHEMES:
		reward_all[scheme] = []

	for l in time_all[SCHEMES[0]]:
		schemes_check = True
		for scheme in SCHEMES:
			if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
				schemes_check = False
				break
		if schemes_check:
			log_file_all.append(l)
			for scheme in SCHEMES:
				reward_all[scheme].append(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN]))

	mean_rewards = {}
	for scheme in SCHEMES:
		mean_rewards[scheme] = np.mean(reward_all[scheme])

	fig = plt.figure()
	ax = fig.add_subplot(111)

	for scheme in SCHEMES:
		ax.plot(reward_all[scheme])
	
	SCHEMES_REW = []
	for scheme in SCHEMES:
		SCHEMES_REW.append(scheme + ': ' + str(mean_rewards[scheme]))

	colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
	for i,j in enumerate(ax.lines):
		j.set_color(colors[i])

	ax.legend(SCHEMES_REW, loc=4)
	
	plt.ylabel('total reward')
	plt.xlabel('trace index')
	plt.savefig(PLOT_FOLDER+'reward_vs_trace.png')
	#plt.show()

	# ---- ---- ---- ----
	# CDF 
	# ---- ---- ---- ----

	fig = plt.figure()
	ax = fig.add_subplot(111)

	for scheme in SCHEMES:
		values, base = np.histogram(reward_all[scheme], bins=NUM_BINS)
		cumulative = np.cumsum(values)
		ax.plot(base[:-1], cumulative)	

	colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
	for i,j in enumerate(ax.lines):
		j.set_color(colors[i])	

	ax.legend(SCHEMES_REW, loc=4)
	
	plt.ylabel('CDF')
	plt.xlabel('total reward')
	plt.savefig(PLOT_FOLDER+'cdf_vs_reward.png')
	#plt.show()


	# ---- ---- ---- ----
	# check each trace
	# ---- ---- ---- ----

	for l in time_all[SCHEMES[0]]:
		schemes_check = True
		for scheme in SCHEMES:
			if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
				schemes_check = False
				break
		if schemes_check:
			fig = plt.figure(figsize=(30,20))

			ax = fig.add_subplot(511)
			for scheme in SCHEMES:
				ax.plot(time_all[scheme][l][:VIDEO_LEN], bit_rate_all[scheme][l][:VIDEO_LEN])
			colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
			for i,j in enumerate(ax.lines):
				j.set_color(colors[i])	
			plt.title(l, fontdict=font)
			plt.ylabel('bit rate selection (kbps)', fontdict=font)

			ax = fig.add_subplot(512)
			for scheme in SCHEMES:
				ax.plot(time_all[scheme][l][:VIDEO_LEN], buff_all[scheme][l][:VIDEO_LEN])
			colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
			for i,j in enumerate(ax.lines):
				j.set_color(colors[i])	
			plt.ylabel('buffer size (sec)', fontdict=font)

			ax = fig.add_subplot(513)
			for scheme in SCHEMES:
				ax.plot(time_all[scheme][l][:VIDEO_LEN], pVMAF_all[scheme][l][:VIDEO_LEN])
			colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
			for i,j in enumerate(ax.lines):
				j.set_color(colors[i])	
			plt.ylabel('VMAF Phone', fontdict=font)

			ax = fig.add_subplot(514)
			for scheme in SCHEMES:
				ax.plot(time_all[scheme][l][:VIDEO_LEN], rMVAF_all[scheme][l][:VIDEO_LEN])
			colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
			for i,j in enumerate(ax.lines):
				j.set_color(colors[i])	
			plt.ylabel('VMAF Regular', fontdict=font)

			ax = fig.add_subplot(515)
			for scheme in SCHEMES:
				ax.plot(time_all[scheme][l][:VIDEO_LEN], bw_all[scheme][l][:VIDEO_LEN])
			colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
			for i,j in enumerate(ax.lines):
				j.set_color(colors[i])	
			plt.ylabel('bandwidth (mbps)', fontdict=font)
			plt.xlabel('time (sec)', fontdict=font)

			SCHEMES_REW = []
			for scheme in SCHEMES:
				SCHEMES_REW.append(scheme + ': ' + str(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN])))

			ax.legend(SCHEMES_REW, loc='best', bbox_to_anchor=(0.5, -0.1), ncol=int(np.ceil(len(SCHEMES) / 2.0)), fontsize='xx-large')
			#plt.show()
			plt.savefig(PLOT_FOLDER+l+'.png')
			plt.close()


if __name__ == '__main__':
	main()
