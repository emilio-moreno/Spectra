from uncertainties import unumpy as un
from uncertainties import ufloat
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import pandas as pd
import glob
from math import isnan


def calculate_angle(degrees, minutes):
	return degrees + minutes / 60


def calculate_refrac_n(minimum_deviation):
	return un.sin((un.radians(prism_angle) + un.radians(minimum_deviation)) / 2) / un.sin(un.radians(prism_angle) / 2)


def cauchy_refrac_n(wl, A, B):
	return A + B / wl**2


class Element():

	def __init__(self, symbol):
		self.symbol = symbol
		self.wavelengths = []
		self.angles = []
		self.refrac_n = []


	def __repr__(self):
		return f"Element symbol: {self.symbol}" \
		f"\nWavelengths of main spectral lines: {self.wavelengths}" \
		f"\nMinimum deviation angles for spectral lines: {self.angles}" \
		f"\nRefractive index: {self.refrac_n}\n"


# Angle error
angle_s = 1 / 120
# Wavelength error is negligible
# as they contain seven significant figures.
wl_s = 0

# Prism apex angle
prism_df = pd.read_csv('./prism_apex.txt')
right_angle1 = calculate_angle(prism_df['degrees'].iloc[0], prism_df['minutes'].iloc[0])
right_angle2 = calculate_angle(prism_df['degrees'].iloc[1], prism_df['minutes'].iloc[1])
left_angle1 = calculate_angle(prism_df['degrees'].iloc[2], prism_df['minutes'].iloc[2])
left_angle2 = calculate_angle(prism_df['degrees'].iloc[3], prism_df['minutes'].iloc[3])
right_average = (ufloat(right_angle1, angle_s) + ufloat(right_angle2, angle_s)) / 2
left_average = (ufloat(left_angle1, angle_s) + ufloat(left_angle2, angle_s)) / 2
# Prism angle is half of the angle between the two reflected rays.
prism_angle = un.fabs(right_average - left_average) / 2
print(f'Prism angle: {prism_angle}\n')

# Listdir for elements
listdir = glob.glob('./*_angles.txt')

elements = []
for filename in listdir:
	element_df = pd.read_csv(filename)
	element_name = filename.replace('_angles.txt', '').replace('.\\', '')
	element = Element(element_name)

	zero_angle = calculate_angle(element_df['degrees'].iloc[0], element_df['minutes'].iloc[0])
	zero_angle = ufloat(zero_angle, angle_s)
	for wl, d, m in zip(element_df['wavelength'][1:], element_df['degrees'][1:], element_df['minutes'][1:]):
		angle = ufloat(calculate_angle(d, m), angle_s)
		angle = un.fabs(angle - zero_angle)
		# We actually measured the angle from
		# the normal to the surface where the
		# light impinges. On minimum deviation
		# the following expression gives the
		# minimum deviation angle.
		min_angle = prism_angle - 2 * angle
		# If there's a repeated value of wl,
		# we're measuring the angle deviation on the left
		# and right edges. So we'll average out.
		if wl in element.wavelengths:
			element.angles[-1] = (element.angles[-1] + min_angle) / 2
			continue
		element.wavelengths.append(wl)
		element.angles.append(min_angle)

	elements.append(element)

for e in elements:
	for wl, angle in zip(e.wavelengths, e.angles):
		e.refrac_n.append(calculate_refrac_n(angle))


# Plots
# Format
colors = ['#3FF', '#F00']
min_wl, max_wl = 380, 700
rc_update = {'font.size': 18, 'font.family': 'serif',
			 'font.serif': ['Times New Roman', 'FreeSerif'],
			 'mathtext.fontset': 'cm'}
plt.rcParams.update(rc_update)

fig, ax = plt.subplots(figsize = (10, 9), dpi = 300)
# We'll append values for a global fit.
global_refrac_ns_n = []
global_wavelengths = []
# I'll exclude sodium as we only have one data point.
for e, c in zip(elements[:-1], colors):
	wavelengths = e.wavelengths
	wls = np.linspace(min_wl, max_wl, 5000)
	refrac_ns_n = [r_n.n for r_n in e.refrac_n]
	refrac_ns_s = [r_n.s for r_n in e.refrac_n]
	global_refrac_ns_n += refrac_ns_n
	global_wavelengths += wavelengths
	popt, pcov = sp.optimize.curve_fit(cauchy_refrac_n, wavelengths, refrac_ns_n)

	# Plotting
	ax.scatter(wavelengths, refrac_ns_n, label = e.symbol, color = c, s = 10)
	ax.errorbar(wavelengths, refrac_ns_n, refrac_ns_s, ls = 'none',
				color = c, capsize = 5)
	ax.plot(wls, cauchy_refrac_n(wls, popt[0], popt[1]),
			label = f'Fit - {e.symbol}', color = c, alpha = 0.45)

	print(e)
	print(f'Fit parameters:')
	print(f'A = {popt[0]:.2e},', f'A_rel_STD = {np.sqrt(pcov[0][0]) * 100 / popt[0]:.2e}%')
	print(f'B = {popt[1]:.2e},', f'B_rel_STD = {np.sqrt(pcov[1][1]) * 100 / popt[1]:.2e}%\n')



# Let's also perform a global fit on Hg and Cd.
popt, pcov = sp.optimize.curve_fit(cauchy_refrac_n, global_wavelengths, global_refrac_ns_n)
ax.plot(wls, cauchy_refrac_n(wls, popt[0], popt[1]),
			label = 'Global fit', color ='k' , alpha = 0.45)

print(f'Combined wavelengths: {global_wavelengths}')
print(f'Global fit parameters:')
print(f'A = {popt[0]:.2e},', f'A_rel_STD = {np.sqrt(pcov[0][0]) * 100/ popt[0]:.2e}%')
print(f'B = {popt[1]:.2e},', f'B_rel_STD = {np.sqrt(pcov[1][1]) * 100/ popt[1]:.2e}%\n')


# Comparison with literature
filename = 'literature/N-BK7.CSV'
df = pd.read_csv(filename)
df = df[df['wl'] >= min_wl / 1000][df['wl'] <= max_wl / 1000]
# Literature is in um, so we multiply by 1000.
wl = df['wl'] * 1000
n = df['n']

# Parameter fit for literature
popt, pcov = sp.optimize.curve_fit(cauchy_refrac_n, wl, n)
print(f'A_lit = {popt[0]:.2e},', f'A_rel_STD = {np.sqrt(pcov[0][0]) * 100/ popt[0]:.2e}%')
print(f'B_lit = {popt[1]:.2e},', f'B_rel_STD = {np.sqrt(pcov[1][1]) * 100/ popt[1]:.2e}%\n')


# Plots
ax.plot(wl, n, color="Green",
		label=f"Literatura: {filename.replace('.csv', '').replace('literature/', '')}")


# Format
ax.set(xlabel = 'Longitud de onda (nm)', ylabel = 'Índice de refracción',
		   title = "Índice de refracción de N-BK7 vs Longitud de onda\n" \
		   "Ajuste: $n(\\lambda) = A + \\frac{B}{\\lambda^2}$")
ax.legend(loc = 1, fontsize = 10)
ax.grid(True, color='#999', linestyle = '--')

plt.savefig('./N-BK7_n_vs_wl_lit.pdf')
# plt.show()
