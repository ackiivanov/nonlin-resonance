import numpy as np
import scipy.integrate as integ
import matplotlib.pyplot as plt
from matplotlib import rc


def find_response(gamma, psi_0, omega, u0, n=150, eps=10**(-4),
				  N=5*10**5, p=1, plot=False):
	r"""
	find_response solves the differential equation
	\ddot{\theta}+2\gamma\dot{\theta}+\sin(\theta)=\psi_0\sin(\omega \tau)
	and returns the amplitude of the oscillation due to driving, if periodic.
	
	Input:
	gamma - damping coefficient in differential equation
	psi_0 - driver amplitude
	omega - driver frequency
	u0 - vector of inital conditions for solution
	n (=150) - number of integration steps per period
	eps (=10**(-4))- stopping tolerance between the last two i-periods
	N (=5*10**5)- maximum steps for integration
	p (=1) - number of fractional frequencies to consider for convergence
	plot (=False) - plots the found solution if True

	Output:
	Amplitude of oscillation if converging to periodic motion with period
	\frac{2 \pi n}{\omega} for n in range(1, p+1), otherwise np.nan,
	which is ignored when plotting
	"""


	# Differential equation in the form \dot{u} = deriv(u, t)
	def deriv(u, t):
		return [u[1],
		        -2*gamma*u[1] - np.sin(u[0]) + psi_0*np.sin(omega*t)]


	# Initialize time range and solution arrays
	ts = np.array([0])
	us = np.array([u0])


	# Print parameters
	print("Solving for gamma={:.3f}, psi_0={:.3f}, omega={:.3f}, "
		  "u_0=[{:.3f}, {:.3f}]".format(gamma, psi_0, omega, *u0))


	# Integrate equation until it settles into oscillation
	# with a frequency \omega / i for i in range(1, p+1)
	osc_offset = [float('inf') for _ in range(p)]
	while min(osc_offset) > eps and len(ts) <= N:
		# Integrate a period over n steps
		tsp = np.linspace(ts[-1], ts[-1] + 2*np.pi/omega, n)
		usp = integ.odeint(deriv, us[-1], tsp)

		# Add new period to the full solution
		ts = np.append(ts[:-1], tsp)
		us = np.append(us[:-1], usp, axis=0)

		# Compare last two i-periods
		for i in range(1, p+1):
			if len(ts) > 2*i*n-1:
				osc_offset[i-1] = sum([abs(us[j-2*i*n+1][0]-us[j-i*n][0])
								for j in range(i*n)])/(i*n)

		# Progress bar (69 can be whatever integer)
		print("#"*(69*len(ts)//N), end="\r", flush=True)

		
	# Extract position from solution
	thetas = np.transpose(us)[0]


	# Plot solution if option enabled
	if plot:
		plt.title(r'$(\gamma={}, \psi_0={}, \omega={:.2f}, u_0={})$'
			       .format(gamma, psi_0, omega, u0))
		plt.ylabel(r'Angle $\theta (\tau)$')
		plt.xlabel(r'Time $\tau$')
	
		plt.plot(ts, thetas)		
		plt.grid()
		plt.show()


	# Report problem when while exited by length restriction
	if len(thetas) > N:
		print('Solution did not converge to an oscillation with angular'
			  ' frequency {} after {} steps'.format(omega, N))
		print('Offset on last period: {}'.format(osc_offset))

		return np.nan


	# Find the absolute extrema
	max_val = max(thetas[-(n*(min(range(p), key=osc_offset.__getitem__)+1)):])
	min_val = min(thetas[-(n*(min(range(p), key=osc_offset.__getitem__)+1)):])	

	return abs(max_val - min_val)/2




# Plotting setup (font size 25 for figures in pdf file)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)


# Parameters for calculation
gamma = 0.9
psi_0 = [0.1, 0.3, 0.5, 0.7] 
omega_min = 0.01
omega_max = 2.01
omega_points = 500
u0 = [0, 0]


r"""
-\gamma_{\mathrm{crit.}}(\psi_0=1.0)=0.56926 \pm 0.00001
-transition happens at \psi_0=1.0
"""

# Make omega, amplitude and outlier arrays
omega = np.linspace(omega_min, omega_max, omega_points)
amp = [[] for _ in range(len(psi_0))]
outl = []


# Calculation for different frequencies
for i in range(len(psi_0)):
	for x in omega:
		ret = find_response(gamma, psi_0[i], x, u0)
		amp[i].append(ret)
		if np.nan_to_num(ret) == 0:
			outl.append(x)


# Plotting
plt.title(r'Resonance curve for $\gamma={}$'.format(gamma))
plt.ylabel(r'Response $\rho (\omega)$')
plt.xlabel(r'Frequency $\omega$')

for i in range(len(amp)):
	plt.plot(omega, amp[i], label=r'$\psi_0 = {}$'.format(psi_0[i]))

for x in outl:
	plt.axvline(x, color='r')

# linear prediction
plt.plot(omega, [psi_0/(((1-x**2)**2 + (2*gamma*x)**2)**(1/2)) for x in omega],
		 'k--', label='Linear prediction')

# asymptotic prediction
plt.plot(omega, [psi_0/(x*(x**2 + 4*gamma**2)**(1/2)) for x in omega],
		 'k--', label='Asymptotic prediction')

plt.legend()
plt.grid()
plt.show()
plt.close()