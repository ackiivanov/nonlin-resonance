import numpy as np
import scipy.integrate as integ
import scipy.signal as sig
import matplotlib.pyplot as plt


def find_response(gamma, omega_0, psi_0, omega, u0, n=20, eps=10**(-4),
				  N=10**5, plot=False):
	r"""
	find_response solves the differential equation
	\ddot{\theta}+2\gamma\dot{\theta}+\omega_0^2\theta=\psi_0\sin(\omega t)
	and returns the amplitude of the oscillation due to driving, if periodic.
	
	Input:
	gamma - damping coefficient in differential equation
	omega_0 - natural oscillation frequency
	psi_0 - driver amplitude
	omega - driver frequency
	u0 - vector of inital conditions for solution
	n (=20) - number of integration steps per period
	eps (=10**(-4))- stopping difference between the last two periods
	N (=10**5)- maximum steps for integration

	Output:
	Amplitude of oscillation if converging to periodic motion with period
	\frac{2\pi}{\omega}, otherwise np.nan, which is ignored when plotting
	"""


	# Differential equation in the form \dot{u} = deriv(u, t)
	def deriv(u, t):
		return [u[1],
		        -2*gamma*u[1] - omega_0**2*np.sin(u[0]) + psi_0*np.sin(omega*t)]


	# Initialize time range and solution arrays
	ts = np.array([0])
	us = np.array([u0])


	# Print parameters
	print("Solving for gamma={:.3f}, omega_0={:.3f}, psi_0={:.3f}, "
		  "omega={:.3f}, u_0=[{:.3f}, {:.3f}]"
		  .format(gamma, omega_0, psi_0, omega, *u0))


	# Integrate equation until it settles into oscillation
	osc_offset = float('inf')
	while osc_offset > eps and len(ts) <= N:
		# Integrate a period over n steps
		tsp = np.linspace(ts[-1], ts[-1] + 2*np.pi/omega, n)
		usp = integ.odeint(deriv, us[-1], tsp) # maybe use solve_ivp

		# Compare current and last period 
		if len(ts) > 1:
			osc_offset = sum([abs(usp[i][0] - us[i-n][0]) for i in range(n)])
			
			# Progress bar (84 can be whatever integer)
			print("#"*(84*len(ts)//N), end="\r", flush=True)

		# Add period to the full solution
		ts = np.append(ts[:-1], tsp)
		us = np.append(us[:-1], usp, axis=0)


	# Extract position from solution
	thetas = np.transpose(us)[0]


	# Plot solution if option enabled (WIP)
	if plot:
		plt.plot(ts, thetas)
		plt.title(r'$(\gamma={}, \omega_0={}, \psi_0={}, \omega={:.2f}, '
			      'u_0={})$'.format(gamma, omega_0, psi_0, omega, u0))
		plt.grid()
		plt.show()


	# Report problem when while exited by length restriction
	if len(thetas) > N:
		print('Solution did not converge to an oscillation with angular'
			  ' frequency {} after {} steps'.format(omega, N))

		return np.nan


	# Find the absolute extrema
	max_val = max(thetas[-n:])
	min_val = min(thetas[-n:])	

	return abs(max_val - min_val)/2


# Parameters for calculation
gamma = 0.2
omega_0 = 1
psi_0 = 2 #0.49
omega_min = 0.01
omega_max = 2.01
omega = np.linspace(omega_min, omega_max, 500)


# Calculation for different frequencies
amp = []
outl = []
for x in omega:
	if x > 0.5 and x < 0.5: #0.44
		ret = find_response(gamma, omega_0, psi_0, x, [0, 0], plot=True)
		amp.append(ret)
		if np.nan_to_num(ret) == 0:
			outl.append(x)
	else:
		ret = find_response(gamma, omega_0, psi_0, x, [0, 0])
		amp.append(ret)
		if np.nan_to_num(ret) == 0:
			outl.append(x)
print(outl)

# Plotting
for x in outl:
	plt.axvline(x, color='r')

plt.plot(omega, amp, 'b-')
plt.plot(omega, [psi_0/(x*(x**2 + 4*gamma**2)**(1/2)) for x in omega], 'g-')
plt.grid()
plt.show()
