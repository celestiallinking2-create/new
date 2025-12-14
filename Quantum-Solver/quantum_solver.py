import math
import numpy as np
from scipy import constants
from scipy.integrate import odeint
from scipy.special import hermite
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EQUATIONS = {
    'particle_in_box': {
        'name': 'Particle in a Box (Infinite Square Well)',
        'description': 'A quantum particle confined to a one-dimensional box with infinitely high walls.',
        'formula': 'ψₙ(x) = √(2/L) sin(nπx/L)',
        'parameters': [
            {'name': 'L', 'label': 'Box Length (nm)', 'default': 1.0, 'min': 0.1, 'max': 10.0},
            {'name': 'n', 'label': 'Quantum Number n', 'default': 1, 'min': 1, 'max': 10, 'type': 'int'},
            {'name': 'm', 'label': 'Particle Mass (electron masses)', 'default': 1.0, 'min': 0.1, 'max': 100.0}
        ]
    },
    'harmonic_oscillator': {
        'name': 'Quantum Harmonic Oscillator',
        'description': 'A particle in a parabolic potential well, fundamental in quantum mechanics.',
        'formula': 'ψₙ(x) = (mω/πℏ)^(1/4) × (1/√(2ⁿn!)) × Hₙ(ξ) × e^(-ξ²/2)',
        'parameters': [
            {'name': 'omega', 'label': 'Angular Frequency ω (10¹⁴ rad/s)', 'default': 1.0, 'min': 0.1, 'max': 10.0},
            {'name': 'n', 'label': 'Quantum Number n', 'default': 0, 'min': 0, 'max': 10, 'type': 'int'},
            {'name': 'm', 'label': 'Particle Mass (electron masses)', 'default': 1.0, 'min': 0.1, 'max': 100.0}
        ]
    },
    'hydrogen_radial': {
        'name': 'Hydrogen Atom Radial Function',
        'description': 'The radial probability distribution for electron in hydrogen atom.',
        'formula': 'P(r) = r² |R_{n,l}(r)|²',
        'parameters': [
            {'name': 'n', 'label': 'Principal Quantum Number n', 'default': 1, 'min': 1, 'max': 5, 'type': 'int'},
            {'name': 'l', 'label': 'Orbital Quantum Number l', 'default': 0, 'min': 0, 'max': 4, 'type': 'int'}
        ]
    },
    'quantum_tunneling': {
        'name': 'Quantum Tunneling',
        'description': 'Probability of a particle tunneling through a potential barrier.',
        'formula': 'T ≈ e^(-2κa) where κ = √(2m(V₀-E))/ℏ',
        'parameters': [
            {'name': 'E', 'label': 'Particle Energy (eV)', 'default': 5.0, 'min': 0.1, 'max': 20.0},
            {'name': 'V0', 'label': 'Barrier Height (eV)', 'default': 10.0, 'min': 1.0, 'max': 50.0},
            {'name': 'a', 'label': 'Barrier Width (nm)', 'default': 0.5, 'min': 0.1, 'max': 5.0},
            {'name': 'm', 'label': 'Particle Mass (electron masses)', 'default': 1.0, 'min': 0.1, 'max': 100.0}
        ]
    },
    'free_particle': {
        'name': 'Free Particle Wave Packet',
        'description': 'A Gaussian wave packet representing a free quantum particle.',
        'formula': 'ψ(x,t) = A × e^(-(x-x₀)²/4σ²) × e^(ikx)',
        'parameters': [
            {'name': 'k', 'label': 'Wave Number k (nm⁻¹)', 'default': 5.0, 'min': 0.1, 'max': 20.0},
            {'name': 'sigma', 'label': 'Wave Packet Width σ (nm)', 'default': 1.0, 'min': 0.1, 'max': 5.0},
            {'name': 'x0', 'label': 'Initial Position x₀ (nm)', 'default': 0.0, 'min': -10.0, 'max': 10.0}
        ]
    }
}


def solve_particle_in_box(params):
    L = params['L'] * 1e-9
    n = int(params['n'])
    m = params['m'] * constants.m_e
    
    hbar = constants.hbar
    E_n = (n**2 * np.pi**2 * hbar**2) / (2 * m * L**2)
    E_eV = E_n / constants.eV
    
    x = np.linspace(0, L, 500)
    psi = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
    prob_density = psi**2
    
    steps = [
        f"Step 1: Set up the infinite square well potential with length L = {params['L']} nm",
        f"Step 2: Apply boundary conditions: ψ(0) = ψ(L) = 0",
        f"Step 3: Solve the time-independent Schrödinger equation: -ℏ²/2m × d²ψ/dx² = Eψ",
        f"Step 4: Apply normalization condition: ∫|ψ|²dx = 1",
        f"Step 5: Energy eigenvalue for n={n}: E_n = n²π²ℏ²/(2mL²) = {E_eV:.6f} eV",
        f"Step 6: Normalized wavefunction: ψ_n(x) = √(2/L) × sin(nπx/L)"
    ]
    
    solution = {
        'energy_eV': E_eV,
        'energy_J': E_n,
        'wavelength_nm': 2*params['L']/n,
        'nodes': n - 1
    }
    
    graph_data = {
        'x': (x * 1e9).tolist(),
        'psi': psi.tolist(),
        'prob': prob_density.tolist(),
        'x_label': 'Position (nm)',
        'y_label': 'Wavefunction ψ(x)',
        'title': f'Particle in a Box: n = {n}'
    }
    
    return solution, steps, graph_data


def solve_harmonic_oscillator(params):
    omega = params['omega'] * 1e14
    n = int(params['n'])
    m = params['m'] * constants.m_e
    
    hbar = constants.hbar
    E_n = hbar * omega * (n + 0.5)
    E_eV = E_n / constants.eV
    
    alpha = np.sqrt(m * omega / hbar)
    x_max = 5 / alpha
    x = np.linspace(-x_max, x_max, 500)
    xi = alpha * x
    
    H_n = hermite(n)
    normalization = (m * omega / (np.pi * hbar))**0.25 / np.sqrt(2**n * math.factorial(n))
    psi = normalization * H_n(xi) * np.exp(-xi**2 / 2)
    prob_density = psi**2
    
    steps = [
        f"Step 1: Set up the harmonic potential V(x) = ½mω²x²",
        f"Step 2: Define characteristic length scale: α = √(mω/ℏ) = {alpha:.4e} m⁻¹",
        f"Step 3: Solve the Schrödinger equation using Hermite polynomials",
        f"Step 4: Energy eigenvalue for n={n}: E_n = ℏω(n + ½) = {E_eV:.6f} eV",
        f"Step 5: Wavefunction involves Hermite polynomial H_{n}(αx)",
        f"Step 6: Zero-point energy E_0 = ½ℏω = {(0.5 * hbar * omega / constants.eV):.6f} eV"
    ]
    
    solution = {
        'energy_eV': E_eV,
        'energy_J': E_n,
        'zero_point_energy_eV': 0.5 * hbar * omega / constants.eV,
        'classical_turning_point_nm': np.sqrt(2*E_n/(m*omega**2)) * 1e9
    }
    
    graph_data = {
        'x': (x * 1e9).tolist(),
        'psi': psi.tolist(),
        'prob': prob_density.tolist(),
        'x_label': 'Position (nm)',
        'y_label': 'Wavefunction ψ(x)',
        'title': f'Quantum Harmonic Oscillator: n = {n}'
    }
    
    return solution, steps, graph_data


def laguerre_poly(n, alpha, x):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 1 + alpha - x
    else:
        L_prev = np.ones_like(x)
        L_curr = 1 + alpha - x
        for k in range(2, n + 1):
            L_next = ((2*k - 1 + alpha - x) * L_curr - (k - 1 + alpha) * L_prev) / k
            L_prev = L_curr
            L_curr = L_next
        return L_curr


def solve_hydrogen_radial(params):
    n = int(params['n'])
    l = int(params['l'])
    
    if l >= n:
        return {'error': f'l must be less than n. Got l={l}, n={n}'}, [], {}
    
    a0 = constants.value('Bohr radius')
    E_n = -13.6 / n**2
    
    r_max = 30 * n**2 * a0
    r = np.linspace(0.001 * a0, r_max, 500)
    rho = 2 * r / (n * a0)
    
    norm_factor = np.sqrt((2/(n*a0))**3 * math.factorial(n-l-1) / (2*n*math.factorial(n+l)))
    L = laguerre_poly(n-l-1, 2*l+1, rho)
    R = norm_factor * np.exp(-rho/2) * rho**l * L
    prob_density = r**2 * R**2
    
    steps = [
        f"Step 1: Set up the Coulomb potential V(r) = -e²/(4πε₀r)",
        f"Step 2: Separate the Schrödinger equation into radial and angular parts",
        f"Step 3: For n={n}, l={l}: Apply boundary conditions at r=0 and r→∞",
        f"Step 4: Energy eigenvalue: E_n = -13.6/n² = {E_n:.4f} eV",
        f"Step 5: Radial function involves associated Laguerre polynomial L_{n-l-1}^(2l+1)",
        f"Step 6: Most probable radius for 1s: r = a₀ = {a0*1e9:.4f} nm"
    ]
    
    r_prob_max_idx = np.argmax(prob_density)
    r_most_probable = r[r_prob_max_idx]
    
    solution = {
        'energy_eV': E_n,
        'bohr_radius_nm': a0 * 1e9,
        'most_probable_radius_nm': r_most_probable * 1e9,
        'orbital': f'{n}{"spdfg"[l] if l < 5 else "?"}'
    }
    
    graph_data = {
        'x': (r * 1e9).tolist(),
        'psi': R.tolist(),
        'prob': prob_density.tolist(),
        'x_label': 'Radius (nm)',
        'y_label': 'Radial Probability Density',
        'title': f'Hydrogen Atom: n={n}, l={l} ({solution["orbital"]} orbital)'
    }
    
    return solution, steps, graph_data


def solve_quantum_tunneling(params):
    E = params['E'] * constants.eV
    V0 = params['V0'] * constants.eV
    a = params['a'] * 1e-9
    m = params['m'] * constants.m_e
    
    hbar = constants.hbar
    
    kappa = 0.0
    if E >= V0:
        T = 1.0
        steps = [
            f"Step 1: Particle energy E = {params['E']} eV",
            f"Step 2: Barrier height V₀ = {params['V0']} eV",
            f"Step 3: Since E ≥ V₀, the particle passes over the barrier classically",
            f"Step 4: Transmission coefficient T = 1 (100%)"
        ]
    else:
        kappa = np.sqrt(2 * m * (V0 - E)) / hbar
        T = np.exp(-2 * kappa * a)
        
        steps = [
            f"Step 1: Set up rectangular barrier with V₀ = {params['V0']} eV, width a = {params['a']} nm",
            f"Step 2: Particle energy E = {params['E']} eV < V₀ (tunneling regime)",
            f"Step 3: Calculate decay constant κ = √(2m(V₀-E))/ℏ = {kappa:.4e} m⁻¹",
            f"Step 4: Apply WKB approximation for transmission",
            f"Step 5: T ≈ e^(-2κa) = {T:.6e}",
            f"Step 6: Tunneling probability: {T*100:.6f}%"
        ]
    
    x = np.linspace(-2*a, 3*a, 500)
    psi_incident = np.ones_like(x) * (x < 0)
    psi_barrier = np.exp(-np.sqrt(2*m*(V0-E))/hbar * (x - 0)) * ((x >= 0) & (x <= a)) if E < V0 else np.ones_like(x) * ((x >= 0) & (x <= a))
    psi_transmitted = T * np.ones_like(x) * (x > a)
    psi_total = psi_incident + psi_barrier + psi_transmitted
    
    V = np.zeros_like(x)
    V[(x >= 0) & (x <= a)] = params['V0']
    
    solution = {
        'transmission_coefficient': T,
        'transmission_percent': T * 100,
        'reflection_coefficient': 1 - T,
        'decay_constant': kappa if E < V0 else 0,
        'penetration_depth_nm': 1/(2*kappa) * 1e9 if E < V0 and kappa > 0 else float('inf')
    }
    
    graph_data = {
        'x': (x * 1e9).tolist(),
        'psi': psi_total.tolist(),
        'prob': V.tolist(),
        'x_label': 'Position (nm)',
        'y_label': 'Wave Amplitude / Potential (eV)',
        'title': f'Quantum Tunneling: T = {T*100:.4f}%'
    }
    
    return solution, steps, graph_data


def solve_free_particle(params):
    k = params['k'] * 1e9
    sigma = params['sigma'] * 1e-9
    x0 = params['x0'] * 1e-9
    
    hbar = constants.hbar
    m = constants.m_e
    
    E = hbar**2 * k**2 / (2 * m)
    E_eV = E / constants.eV
    wavelength = 2 * np.pi / k
    
    x_range = 5 * sigma
    x = np.linspace(x0 - x_range, x0 + x_range, 500)
    
    A = 1 / (2 * np.pi * sigma**2)**0.25
    psi_real = A * np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.cos(k * x)
    psi_imag = A * np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.sin(k * x)
    psi_mag = A * np.exp(-(x - x0)**2 / (4 * sigma**2))
    prob_density = psi_mag**2
    
    steps = [
        f"Step 1: Construct Gaussian wave packet centered at x₀ = {params['x0']} nm",
        f"Step 2: Wave packet width σ = {params['sigma']} nm",
        f"Step 3: Central wave number k = {params['k']} nm⁻¹",
        f"Step 4: de Broglie wavelength λ = 2π/k = {wavelength*1e9:.4f} nm",
        f"Step 5: Kinetic energy E = ℏ²k²/(2m) = {E_eV:.6f} eV",
        f"Step 6: Momentum uncertainty Δp ≈ ℏ/(2σ)"
    ]
    
    solution = {
        'energy_eV': E_eV,
        'wavelength_nm': wavelength * 1e9,
        'momentum': hbar * k,
        'velocity': hbar * k / m,
        'uncertainty_x_nm': sigma * 1e9,
        'uncertainty_p': hbar / (2 * sigma)
    }
    
    graph_data = {
        'x': (x * 1e9).tolist(),
        'psi': psi_mag.tolist(),
        'prob': prob_density.tolist(),
        'x_label': 'Position (nm)',
        'y_label': 'Wave Packet |ψ(x)|',
        'title': f'Free Particle Wave Packet: k = {params["k"]} nm⁻¹'
    }
    
    return solution, steps, graph_data


def solve_equation(equation_type, params):
    solvers = {
        'particle_in_box': solve_particle_in_box,
        'harmonic_oscillator': solve_harmonic_oscillator,
        'hydrogen_radial': solve_hydrogen_radial,
        'quantum_tunneling': solve_quantum_tunneling,
        'free_particle': solve_free_particle
    }
    
    if equation_type not in solvers:
        return {'error': f'Unknown equation type: {equation_type}'}, [], {}
    
    return solvers[equation_type](params)


def generate_graph(graph_data, save_path=None):
    if not graph_data:
        return None
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.patch.set_facecolor('#1a1a2e')
    
    ax1.set_facecolor('#16213e')
    ax1.plot(graph_data['x'], graph_data['psi'], color='#00d9ff', linewidth=2, label='Wavefunction')
    ax1.set_xlabel(graph_data['x_label'], color='#e0e0e0')
    ax1.set_ylabel('Wavefunction ψ(x)', color='#e0e0e0')
    ax1.set_title(graph_data['title'], color='#ffffff', fontsize=14, fontweight='bold')
    ax1.tick_params(colors='#e0e0e0')
    ax1.grid(True, alpha=0.3, color='#4a4a6a')
    ax1.spines['bottom'].set_color('#4a4a6a')
    ax1.spines['top'].set_color('#4a4a6a')
    ax1.spines['left'].set_color('#4a4a6a')
    ax1.spines['right'].set_color('#4a4a6a')
    
    ax2.set_facecolor('#16213e')
    ax2.fill_between(graph_data['x'], graph_data['prob'], color='#ff6b9d', alpha=0.5)
    ax2.plot(graph_data['x'], graph_data['prob'], color='#ff6b9d', linewidth=2, label='Probability Density')
    ax2.set_xlabel(graph_data['x_label'], color='#e0e0e0')
    ax2.set_ylabel('Probability Density |ψ|²', color='#e0e0e0')
    ax2.tick_params(colors='#e0e0e0')
    ax2.grid(True, alpha=0.3, color='#4a4a6a')
    ax2.spines['bottom'].set_color('#4a4a6a')
    ax2.spines['top'].set_color('#4a4a6a')
    ax2.spines['left'].set_color('#4a4a6a')
    ax2.spines['right'].set_color('#4a4a6a')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='jpg', dpi=150, facecolor='#1a1a2e', edgecolor='none')
        plt.close()
        return save_path
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='#1a1a2e', edgecolor='none')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return img_base64


def get_equations():
    return EQUATIONS
