# @title
from ipywidgets import interact, Dropdown

# Define all codes
def KPP():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import quad
    from ipywidgets import interact, IntSlider, FloatSlider

    # Default parameters
    default_N_ = 10
    N = 2 * default_N_ + 1
    default_a = 1.0
    default_rho = 0.5
    default_V0 = 100
    default_iter = 200
    default_bands = 5
    h_pot = np.zeros((N, N))
    basis_arr = np.array([])

    def calculate_and_plot(N_, a, V0, rho, iter_, bands):
        global default_N_, default_a, default_V0, default_rho, h_pot, basis_arr
        default_N_ = N_
        default_rho = rho
        default_a = a
        default_V0 = V0

        N = 2 * N_ + 1
        h_pot = np.zeros((N, N))

        basis_arr = np.array([np.sum([i * (-1) ** i for i in range(j + 1)]) for j in range(N)])
        seq_arr = np.array([i * (-1) ** i for i in range(N)])

        b = rho * a
        eta = 0.0103198 * V0 * (a ** 2)
        fac = 10 ** (-10)

        def f1(x):
            return 0

        def f2(x):
            return 1

        def g1(x, p, q):
            return np.exp(complex(0, 1) * 2 * np.pi * (q - p) / (a * fac)) * f1(x)

        def g2(x, p, q):
            return np.exp(complex(0, 1) * 2 * np.pi * (q - p) / (a * fac)) * f2(x)

        vpp = (1 / (a * fac)) * (quad(f1, 0 * fac, b * fac)[0] + quad(f2, b * fac, a * fac)[0])
        def vpq(p, q):
            return (1 / (a * fac)) * (
                        quad(g1, 0 * fac, b * fac, args=(p, q))[0] + quad(g2, b * fac, a * fac, args=(p, q))[0])

        for i in range(N):
            for j in range(N):
                if i == j:
                    h_pot[i, j] = eta * vpp
                else:
                    h_pot[i, j] = eta * vpq(i, j)

        E_1 = np.zeros((N, iter_ + 1))
        ka_1 = np.linspace(-np.pi, np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_1[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_1[:, i] = np.sort(np.real(E_eig))

        E_2 = np.zeros((N, iter_ + 1))
        ka_2 = np.linspace(-6*np.pi, 6*np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_2[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_2[:, i] = np.sort(np.real(E_eig))
        E_3 = np.zeros((N, iter_ + 1))
        ka_3 = np.linspace(-bands*np.pi, bands*np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_3[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_3[:, i] = np.sort(np.real(E_eig))

        plt.figure(figsize=(14, 14))
        plt.subplot(2, 2, 1)
        n_values = np.arange(-10, 11)
        def V(x):
            return V0 * sum(np.heaviside(x - (n*a + b), 1) * np.heaviside((n + 1)*a - x, 1) for n in n_values)
        x_pot = np.linspace(-10.1, 10.1, 1000)
        plt.plot(x_pot, V(x_pot))
        plt.title('Periodic Potential Function (PPF) for KPP')
        plt.xlabel('x')
        plt.ylabel('V(x)')

        plt.subplot(2, 2, 2)
        c_map=plt.cm.get_cmap('hsv', bands + 1)
        for i in range(bands):
            plt.plot(ka_1 / np.pi, E_1[i], '.', color=c_map(i), markersize=1)
        for x_coord in [-1, 1]:
            plt.axvline(x=x_coord, color='black', linestyle='--', linewidth=0.5)
        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Reduced Zone Scheme (RZS) for KPP')

        plt.subplot(2, 2, 3)
        for i in range(bands):
            plt.plot(ka_2 / np.pi, E_2[i], '.', color=c_map(i), markersize=1)
        for x_coord in [-6, 6]:
            plt.axvline(x=x_coord, color='black', linestyle='--', linewidth=0.5)
        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Periodic Zone Scheme (PZS) for KPP')

        plt.subplot(2, 2, 4)
        colors = ['k', 'k', 'k', 'k', 'k','k', 'k', 'k', 'k', 'k']
        for i in range(iter_ + 1):
            x_values = ka_3[i] / np.pi
            y_values = E_3[:, i]

            for band, color in zip(range(1, bands+1), colors):
                plt.plot(x_values, np.where(((x_values >= -band) & (x_values <= -band+1)) |
                                            ((x_values >= band-1) & (x_values <= band)),
                                            y_values[band-1], np.nan), color+'.', markersize=1)

        for x_coord in range(-bands, bands+1):
            plt.axvline(x=x_coord, color='brown', linestyle='--', linewidth=0.4)

        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Extended Zone Scheme (EZS) for KPP')
        plt.suptitle('Kroning Penney Potential (KPP)', color='c',fontsize=25)
        plt.figtext(0.5, 0.01, '\u00A9 Designed by Avoy Jana (MSc, IITD)', ha='center', va='top', fontsize=12, color='gray')
        plt.tight_layout()
        plt.show()

    # Interactive widget
    interact(calculate_and_plot,
            N_=IntSlider(min=5, max=100, step=5, value=default_N_, description='Basis limit', continuous_update=False),
            a=FloatSlider(min=0.1, max=10, step=0.1, value=default_a, description='a (A)', continuous_update=False),
            V0=FloatSlider(min=0, max=1000, step=0.1, value=default_V0, description='V0 (eV)', continuous_update=False),
            rho=FloatSlider(min=0, max=1, step=0.1, value=default_rho, description='b/a (rho)', continuous_update=False),
            iter_=IntSlider(min=10, max=1000, step=10, value=default_iter, description='# of iters', continuous_update=False),
            bands=IntSlider(min=1, max=10, value=default_bands, description='# of bands', continuous_update=False))


def PPP():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import quad
    from ipywidgets import interact, IntSlider, FloatSlider

    # Default parameters
    default_N_ = 10
    N = 2 * default_N_ + 1
    default_a = 1.0
    default_rho = 0.5
    default_kpr = 1
    default_iter = 200
    default_bands = 5
    h_pot = np.zeros((N, N))
    basis_arr = np.array([])

    def calculate_and_plot(N_, a, kpr, rho, iter_, bands):
        global default_N_, default_a, default_kpr, default_V0, default_rho, h_pot, basis_arr
        default_N_ = N_
        default_rho = rho
        default_a = a
        default_kpr = kpr

        N = 2 * N_ + 1
        h_pot = np.zeros((N, N))

        basis_arr = np.array([np.sum([i * (-1) ** i for i in range(j + 1)]) for j in range(N)])
        seq_arr = np.array([i * (-1) ** i for i in range(N)])

        b = rho * a
        gamma = 0.582*(kpr**0.5)*(a** 2)
        gn=(np.pi*gamma/2)**2
        fac = 10 ** (-10)

        def f(x):
            return (x/(a*fac)-1/2-rho)**2
        def g(x,p,q):
            return np.exp(complex(0,1)*2*np.pi*(q-p)/(a*fac))*f(x)

        vpp=(1/(a*fac))*quad(f, 0, a*fac)[0]
        def vpq(p, q):
            return (1/(a*fac))*quad(g, 0*fac, a*fac, args=(p, q))[0]

        for i in range(N):
            for j in range(N):
                if i == j:
                    h_pot[i, j] = gn * vpp
                else:
                    h_pot[i, j] = gn * vpq(i, j)

        E_1 = np.zeros((N, iter_ + 1))
        ka_1 = np.linspace(-np.pi, np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_1[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_1[:, i] = np.sort(np.real(E_eig))

        E_2 = np.zeros((N, iter_ + 1))
        ka_2 = np.linspace(-6*np.pi, 6*np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_2[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_2[:, i] = np.sort(np.real(E_eig))
        E_3 = np.zeros((N, iter_ + 1))
        ka_3 = np.linspace(-bands*np.pi, bands*np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_3[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_3[:, i] = np.sort(np.real(E_eig))

        plt.figure(figsize=(14, 14))
        plt.subplot(2, 2, 1)
        n_values = np.arange(-10, 11)
        def parabola(n, x):
            return (1/2)*kpr*(x - ((2*n-1)/2*a+b))**2
        def V(x):
            return sum(parabola(n,x) * (np.heaviside(x - ((n-1)*a+b), 0) - np.heaviside(x - (n*a+b), 0)) for n in n_values)
        x_pot = np.linspace(-10.1, 10.1, 1000)
        plt.plot(x_pot, V(x_pot))
        plt.title('Periodic Potential Function (PPF) for PPP')
        plt.xlabel('x')
        plt.ylabel('V(x)')

        plt.subplot(2, 2, 2)
        c_map=plt.cm.get_cmap('hsv', bands + 1)
        for i in range(bands):
            plt.plot(ka_1 / np.pi, E_1[i], '.', color=c_map(i), markersize=1)
        for x_coord in [-1, 1]:
            plt.axvline(x=x_coord, color='black', linestyle='--', linewidth=0.5)
        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Reduced Zone Scheme (RZS) for PPP')

        plt.subplot(2, 2, 3)
        for i in range(bands):
            plt.plot(ka_2 / np.pi, E_2[i], '.', color=c_map(i), markersize=1)
        for x_coord in [-6, 6]:
            plt.axvline(x=x_coord, color='black', linestyle='--', linewidth=0.5)
        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Periodic Zone Scheme (PZS) for PPP')

        plt.subplot(2, 2, 4)
        colors = ['k', 'k', 'k', 'k', 'k','k', 'k', 'k', 'k', 'k']
        for i in range(iter_ + 1):
            x_values = ka_3[i] / np.pi
            y_values = E_3[:, i]

            for band, color in zip(range(1, bands+1), colors):
                plt.plot(x_values, np.where(((x_values >= -band) & (x_values <= -band+1)) |
                                            ((x_values >= band-1) & (x_values <= band)),
                                            y_values[band-1], np.nan), color+'.', markersize=1)

        for x_coord in range(-bands, bands+1):
            plt.axvline(x=x_coord, color='brown', linestyle='--', linewidth=0.4)

        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Extended Zone Scheme (EZS) for PPP')
        plt.suptitle('Periodic Parabolic Potential (KPP)', color='c',fontsize=25)
        plt.figtext(0.5, 0.01, '\u00A9 Designed by Avoy Jana (MSc, IITD)', ha='center', va='top', fontsize=12, color='gray')
        plt.tight_layout()
        plt.show()

    # Interactive widget
    interact(calculate_and_plot,
            N_=IntSlider(min=5, max=100, step=5, value=default_N_, description='Basis limit', continuous_update=False),
            a=FloatSlider(min=0.1, max=10, step=0.1, value=default_a, description='a (A)', continuous_update=False),
            kpr=FloatSlider(min=1, max=5, step=0.1, value=default_kpr, description='kpr (kN/m)', continuous_update=False),
            rho=FloatSlider(min=0, max=1, step=0.1, value=default_rho, description='b/a (rho)', continuous_update=False),
            iter_=IntSlider(min=10, max=1000, step=10, value=default_iter, description='# of iters', continuous_update=False),
            bands=IntSlider(min=1, max=10, value=default_bands, description='# of bands', continuous_update=False))
def IPPP():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import quad
    from ipywidgets import interact, IntSlider, FloatSlider

    # Default parameters
    default_N_ = 10
    N = 2 * default_N_ + 1
    default_a = 1.0
    default_rho = 0.5
    default_kpr = 1
    default_iter = 200
    default_bands = 5
    h_pot = np.zeros((N, N))
    basis_arr = np.array([])

    def calculate_and_plot(N_, a, kpr, rho, iter_, bands):
        global default_N_, default_a, default_kpr, default_V0, default_rho, h_pot, basis_arr
        default_N_ = N_
        default_rho = rho
        default_a = a
        default_kpr = kpr

        N = 2 * N_ + 1
        h_pot = np.zeros((N, N))

        basis_arr = np.array([np.sum([i * (-1) ** i for i in range(j + 1)]) for j in range(N)])
        seq_arr = np.array([i * (-1) ** i for i in range(N)])

        b = rho * a
        gamma = 0.582*(kpr**0.5)*(a** 2)
        gn=(np.pi*gamma/2)**2
        fac = 10 ** (-10)

        def f(x):
            return (1/4)-(x/(a*fac)-1/2-rho)**2
        def g(x,p,q):
            return np.exp(complex(0,1)*2*np.pi*(q-p)/(a*fac))*f(x)

        vpp=(1/(a*fac))*quad(f, 0, a*fac)[0]
        def vpq(p, q):
            return (1/(a*fac))*quad(g, 0*fac, a*fac, args=(p, q))[0]

        for i in range(N):
            for j in range(N):
                if i == j:
                    h_pot[i, j] = gn * vpp
                else:
                    h_pot[i, j] = gn * vpq(i, j)

        E_1 = np.zeros((N, iter_ + 1))
        ka_1 = np.linspace(-np.pi, np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_1[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_1[:, i] = np.sort(np.real(E_eig))

        E_2 = np.zeros((N, iter_ + 1))
        ka_2 = np.linspace(-6*np.pi, 6*np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_2[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_2[:, i] = np.sort(np.real(E_eig))
        E_3 = np.zeros((N, iter_ + 1))
        ka_3 = np.linspace(-bands*np.pi, bands*np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_3[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_3[:, i] = np.sort(np.real(E_eig))

        plt.figure(figsize=(14, 14))
        plt.subplot(2, 2, 1)
        n_values = np.arange(-10, 11)
        def parabola(n, x):
            return (1/2)*kpr*(a/2)**2-(1/2)*kpr*(x - ((2*n-1)/2*a+b))**2
        def V(x):
            return sum(parabola(n,x) * (np.heaviside(x - ((n-1)*a+b), 0) - np.heaviside(x - (n*a+b), 0)) for n in n_values)
        x_pot = np.linspace(-10.1, 10.1, 1000)
        plt.plot(x_pot, V(x_pot))
        plt.title('Periodic Potential Function (PPF) for IPPP')
        plt.xlabel('x')
        plt.ylabel('V(x)')

        plt.subplot(2, 2, 2)
        c_map=plt.cm.get_cmap('hsv', bands + 1)
        for i in range(bands):
            plt.plot(ka_1 / np.pi, E_1[i], '.', color=c_map(i), markersize=1)
        for x_coord in [-1, 1]:
            plt.axvline(x=x_coord, color='black', linestyle='--', linewidth=0.5)
        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Reduced Zone Scheme (RZS) for IPPP')

        plt.subplot(2, 2, 3)
        for i in range(bands):
            plt.plot(ka_2 / np.pi, E_2[i], '.', color=c_map(i), markersize=1)
        for x_coord in [-6, 6]:
            plt.axvline(x=x_coord, color='black', linestyle='--', linewidth=0.5)
        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Periodic Zone Scheme (PZS) for IPPP')

        plt.subplot(2, 2, 4)
        colors = ['k', 'k', 'k', 'k', 'k','k', 'k', 'k', 'k', 'k']
        for i in range(iter_ + 1):
            x_values = ka_3[i] / np.pi
            y_values = E_3[:, i]

            for band, color in zip(range(1, bands+1), colors):
                plt.plot(x_values, np.where(((x_values >= -band) & (x_values <= -band+1)) |
                                            ((x_values >= band-1) & (x_values <= band)),
                                            y_values[band-1], np.nan), color+'.', markersize=1)

        for x_coord in range(-bands, bands+1):
            plt.axvline(x=x_coord, color='brown', linestyle='--', linewidth=0.4)

        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Extended Zone Scheme (EZS) for IPPP')
        plt.suptitle('Inverse PPP (IPPP)', color='c',fontsize=25)
        plt.figtext(0.5, 0.01, '\u00A9 Designed by Avoy Jana (MSc, IITD)', ha='center', va='top', fontsize=12, color='gray')
        plt.tight_layout()
        plt.show()

    # Interactive widget
    interact(calculate_and_plot,
            N_=IntSlider(min=5, max=100, step=5, value=default_N_, description='Basis limit', continuous_update=False),
            a=FloatSlider(min=0.1, max=10, step=0.1, value=default_a, description='a (A)', continuous_update=False),
            kpr=FloatSlider(min=1, max=5, step=0.1, value=default_kpr, description='kpr (kN/m)', continuous_update=False),
            rho=FloatSlider(min=0, max=1, step=0.1, value=default_rho, description='b/a (rho)', continuous_update=False),
            iter_=IntSlider(min=10, max=1000, step=10, value=default_iter, description='# of iters', continuous_update=False),
            bands=IntSlider(min=1, max=10, value=default_bands, description='# of bands', continuous_update=False))
def PGP():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import quad
    from ipywidgets import interact, IntSlider, FloatSlider

    # Default parameters
    default_N_ = 10
    N = 2 * default_N_ + 1
    default_a = 8.0
    default_rho = 0.5
    default_V0 = 100
    default_sigma = 2.2
    default_iter = 200
    default_bands = 5
    h_pot = np.zeros((N, N))
    basis_arr = np.array([])

    def calculate_and_plot(N_, a, V0, sigma, rho, iter_, bands):
        global default_N_, default_a, default_V0, default_sigma, default_V0, default_rho, h_pot, basis_arr
        default_N_ = N_
        default_rho = rho
        default_a = a
        default_V0 = V0
        default_sigma = sigma

        N = 2 * N_ + 1
        h_pot = np.zeros((N, N))

        basis_arr = np.array([np.sum([i * (-1) ** i for i in range(j + 1)]) for j in range(N)])
        seq_arr = np.array([i * (-1) ** i for i in range(N)])

        b = rho * a
        eta = 0.0103198*V0*(a** 2)
        fac = 10 ** (-10)
        def f(x):
            return np.exp(-(a/(np.sqrt(2)*sigma))**2*(x/a-1/2-rho)**2)
        def g(x,p,q):
            return np.exp(complex(0,1)*2*np.pi*(q-p)/(a*fac))*f(x)

        vpp=(1/(a*fac))*quad(f, 0, a*fac)[0]
        def vpq(p, q):
            return (1/(a*fac))*quad(g, 0*fac, a*fac, args=(p, q))[0]

        for i in range(N):
            for j in range(N):
                if i == j:
                    h_pot[i, j] = eta * vpp
                else:
                    h_pot[i, j] = eta * vpq(i, j)

        E_1 = np.zeros((N, iter_ + 1))
        ka_1 = np.linspace(-np.pi, np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_1[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_1[:, i] = np.sort(np.real(E_eig))

        E_2 = np.zeros((N, iter_ + 1))
        ka_2 = np.linspace(-6*np.pi, 6*np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_2[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_2[:, i] = np.sort(np.real(E_eig))
        E_3 = np.zeros((N, iter_ + 1))
        ka_3 = np.linspace(-bands*np.pi, bands*np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_3[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_3[:, i] = np.sort(np.real(E_eig))

        plt.figure(figsize=(14, 14))
        plt.subplot(2, 2, 1)
        n_values = np.arange(-10, 11)
        def gaussian(n, x):
            return V0*np.exp(-((x - ((2*n-1)/2*a+b))**2)/sigma**2)
        def V(x):
            return sum(gaussian(n,x) * (np.heaviside(x - ((n-1)*a+b), 0) - np.heaviside(x - (n*a+b), 0)) for n in n_values)
        x_pot = np.linspace(-5, 5, 1000)
        plt.plot(x_pot, V(x_pot))
        plt.title('Periodic Potential Function (PPF) for PGP')
        plt.xlabel('x')
        plt.ylabel('V(x)')

        plt.subplot(2, 2, 2)
        c_map=plt.cm.get_cmap('hsv', bands + 1)
        for i in range(bands):
            plt.plot(ka_1 / np.pi, E_1[i], '.', color=c_map(i), markersize=1)
        for x_coord in [-1, 1]:
            plt.axvline(x=x_coord, color='black', linestyle='--', linewidth=0.5)
        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Reduced Zone Scheme (RZS) for PGP')

        plt.subplot(2, 2, 3)
        for i in range(bands):
            plt.plot(ka_2 / np.pi, E_2[i], '.', color=c_map(i), markersize=1)
        for x_coord in [-6, 6]:
            plt.axvline(x=x_coord, color='black', linestyle='--', linewidth=0.5)
        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Periodic Zone Scheme (PZS) for PGP')

        plt.subplot(2, 2, 4)
        colors = ['k', 'k', 'k', 'k', 'k','k', 'k', 'k', 'k', 'k']
        for i in range(iter_ + 1):
            x_values = ka_3[i] / np.pi
            y_values = E_3[:, i]

            for band, color in zip(range(1, bands+1), colors):
                plt.plot(x_values, np.where(((x_values >= -band) & (x_values <= -band+1)) |
                                            ((x_values >= band-1) & (x_values <= band)),
                                            y_values[band-1], np.nan), color+'.', markersize=1)

        for x_coord in range(-bands, bands+1):
            plt.axvline(x=x_coord, color='brown', linestyle='--', linewidth=0.4)

        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Extended Zone Scheme (EZS) for PGP')
        plt.suptitle('Periodic Gaussian Potential (PGP)', color='c',fontsize=25)
        plt.figtext(0.5, 0.01, '\u00A9 Designed by Avoy Jana (MSc, IITD)', ha='center', va='top', fontsize=12, color='gray')
        plt.tight_layout()
        plt.show()

    # Interactive widget
    interact(calculate_and_plot,
            N_=IntSlider(min=5, max=100, step=5, value=default_N_, description='Basis limit', continuous_update=False),
            a=FloatSlider(min=0.1, max=10, step=0.1, value=default_a, description='a (A)', continuous_update=False),
            V0=FloatSlider(min=0, max=1000, step=10, value=default_V0, description='V0 (eV)', continuous_update=False),
            sigma=FloatSlider(min=0.1, max=5, step=0.1, value=default_sigma, description='sigma (A)', continuous_update=False),
            rho=FloatSlider(min=0, max=1, step=0.1, value=default_rho, description='b/a (rho)', continuous_update=False),
            iter_=IntSlider(min=10, max=1000, step=10, value=default_iter, description='# of iters', continuous_update=False),
            bands=IntSlider(min=1, max=10, value=default_bands, description='# of bands', continuous_update=False))
def IPGP():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import quad
    from ipywidgets import interact, IntSlider, FloatSlider

    # Default parameters
    default_N_ = 10
    N = 2 * default_N_ + 1
    default_a = 8.0
    default_rho = 0.5
    default_V0 = 100
    default_sigma = 2.2
    default_iter = 200
    default_bands = 5
    h_pot = np.zeros((N, N))
    basis_arr = np.array([])

    def calculate_and_plot(N_, a, V0, sigma, rho, iter_, bands):
        global default_N_, default_a, default_sigma, default_V0, default_rho, h_pot, basis_arr
        default_N_ = N_
        default_rho = rho
        default_a = a
        default_V0 = V0
        default_sigma = sigma

        N = 2 * N_ + 1
        h_pot = np.zeros((N, N))

        basis_arr = np.array([np.sum([i * (-1) ** i for i in range(j + 1)]) for j in range(N)])
        seq_arr = np.array([i * (-1) ** i for i in range(N)])

        b = rho * a
        eta = 0.0103198*V0*(a** 2)
        fac = 10 ** (-10)
        def f(x):
            return 1-np.exp(-(a/(np.sqrt(2)*sigma))**2*(x/a-1/2-rho)**2)
        def g(x,p,q):
            return np.exp(complex(0,1)*2*np.pi*(q-p)/(a*fac))*f(x)

        vpp=(1/(a*fac))*quad(f, 0, a*fac)[0]
        def vpq(p, q):
            return (1/(a*fac))*quad(g, 0*fac, a*fac, args=(p, q))[0]

        for i in range(N):
            for j in range(N):
                if i == j:
                    h_pot[i, j] = eta * vpp
                else:
                    h_pot[i, j] = eta * vpq(i, j)

        E_1 = np.zeros((N, iter_ + 1))
        ka_1 = np.linspace(-np.pi, np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_1[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_1[:, i] = np.sort(np.real(E_eig))

        E_2 = np.zeros((N, iter_ + 1))
        ka_2 = np.linspace(-6*np.pi, 6*np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_2[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_2[:, i] = np.sort(np.real(E_eig))
        E_3 = np.zeros((N, iter_ + 1))
        ka_3 = np.linspace(-bands*np.pi, bands*np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_3[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_3[:, i] = np.sort(np.real(E_eig))

        plt.figure(figsize=(14, 14))
        plt.subplot(2, 2, 1)
        n_values = np.arange(-10, 11)
        def gaussian(n, x):
            return V0*(1-np.exp(-((x - ((2*n-1)/2*a+b))**2)/sigma**2))
        def V(x):
            return sum(gaussian(n,x) * (np.heaviside(x - ((n-1)*a+b), 0) - np.heaviside(x - (n*a+b), 0)) for n in n_values)
        x_pot = np.linspace(-5, 5, 1000)
        plt.plot(x_pot, V(x_pot))
        plt.title('Periodic Potential Function (PPF) for IPGP')
        plt.xlabel('x')
        plt.ylabel('V(x)')

        plt.subplot(2, 2, 2)
        c_map=plt.cm.get_cmap('hsv', bands + 1)
        for i in range(bands):
            plt.plot(ka_1 / np.pi, E_1[i], '.', color=c_map(i), markersize=1)
        for x_coord in [-1, 1]:
            plt.axvline(x=x_coord, color='black', linestyle='--', linewidth=0.5)
        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Reduced Zone Scheme (RZS) for IPGP')

        plt.subplot(2, 2, 3)
        for i in range(bands):
            plt.plot(ka_2 / np.pi, E_2[i], '.', color=c_map(i), markersize=1)
        for x_coord in [-6, 6]:
            plt.axvline(x=x_coord, color='black', linestyle='--', linewidth=0.5)
        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Periodic Zone Scheme (PZS) for IPGP')

        plt.subplot(2, 2, 4)
        colors = ['k', 'k', 'k', 'k', 'k','k', 'k', 'k', 'k', 'k']
        for i in range(iter_ + 1):
            x_values = ka_3[i] / np.pi
            y_values = E_3[:, i]

            for band, color in zip(range(1, bands+1), colors):
                plt.plot(x_values, np.where(((x_values >= -band) & (x_values <= -band+1)) |
                                            ((x_values >= band-1) & (x_values <= band)),
                                            y_values[band-1], np.nan), color+'.', markersize=1)

        for x_coord in range(-bands, bands+1):
            plt.axvline(x=x_coord, color='brown', linestyle='--', linewidth=0.4)

        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Extended Zone Scheme (EZS) for IPGP')
        plt.suptitle('Inverse PGP (IPGP)', color='c',fontsize=25)
        plt.figtext(0.5, 0.01, '\u00A9 Designed by Avoy Jana (MSc, IITD)', ha='center', va='top', fontsize=12, color='gray')
        plt.tight_layout()
        plt.show()

    # Interactive widget
    interact(calculate_and_plot,
            N_=IntSlider(min=5, max=100, step=5, value=default_N_, description='Basis limit', continuous_update=False),
            a=FloatSlider(min=0.1, max=10, step=0.1, value=default_a, description='a (A)', continuous_update=False),
            V0=FloatSlider(min=0, max=1000, step=10, value=default_V0, description='V0 (eV)', continuous_update=False),
            sigma=FloatSlider(min=0.1, max=5, step=0.1, value=default_sigma, description='sigma (A)', continuous_update=False),
            rho=FloatSlider(min=0, max=1, step=0.1, value=default_rho, description='b/a (rho)', continuous_update=False),
            iter_=IntSlider(min=10, max=1000, step=10, value=default_iter, description='# of iters', continuous_update=False),
            bands=IntSlider(min=1, max=10, value=default_bands, description='# of bands', continuous_update=False))
def PLP():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import quad
    from ipywidgets import interact, IntSlider, FloatSlider

    # Default parameters
    default_N_ = 10
    N = 2 * default_N_ + 1
    default_a = 1.0
    default_rho = 0.5
    default_V0 = 100
    default_iter = 200
    default_bands = 5
    h_pot = np.zeros((N, N))
    basis_arr = np.array([])

    def calculate_and_plot(N_, a, V0, rho, iter_, bands):
        global default_N_, default_a, default_V0, default_rho, h_pot, basis_arr
        default_N_ = N_
        default_rho = rho
        default_a = a
        default_V0 = V0

        N = 2 * N_ + 1
        h_pot = np.zeros((N, N))

        basis_arr = np.array([np.sum([i * (-1) ** i for i in range(j + 1)]) for j in range(N)])
        seq_arr = np.array([i * (-1) ** i for i in range(N)])

        b = rho * a
        eta = 0.0103198*V0*(a** 2)
        fac = 10 ** (-10)

        def f1(x):
          return 2*x/(a*fac)-2*rho-1
        def f2(x):
          return -f1(x)
        def g1(x,p,q):
          return np.exp(complex(0,1)*2*np.pi*(q-p)/(a*fac))*f1(x)
        def g2(x,p,q):
          return np.exp(complex(0,1)*2*np.pi*(q-p)/(a*fac))*f2(x)

        vpp=quad(f1, b*fac, (a/2+b)*fac)[0]+quad(f2,(a/2+b)*fac,(a+b)*fac)[0]
        def vpq(p, q):
          return quad(g1, b*fac, (a/2+b)*fac, args=(p, q))[0]+quad(g2, (a/2+b)*fac, (a+b)*fac, args=(p, q))[0]

        for i in range(N):
            for j in range(N):
                if i == j:
                    h_pot[i, j] = eta * vpp
                else:
                    h_pot[i, j] = eta * vpq(i, j)

        E_1 = np.zeros((N, iter_ + 1))
        ka_1 = np.linspace(-np.pi, np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_1[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_1[:, i] = np.sort(np.real(E_eig))

        E_2 = np.zeros((N, iter_ + 1))
        ka_2 = np.linspace(-6*np.pi, 6*np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_2[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_2[:, i] = np.sort(np.real(E_eig))
        E_3 = np.zeros((N, iter_ + 1))
        ka_3 = np.linspace(-bands*np.pi, bands*np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_3[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_3[:, i] = np.sort(np.real(E_eig))

        plt.figure(figsize=(14, 14))
        plt.subplot(2, 2, 1)
        n_values = np.arange(-10, 11)
        def triangular(n, x):
            return V0 * np.abs((x -((n-1)*a+b)) / (a/2) -1) * (-np.heaviside(x -(n*a+b), 0) + np.heaviside(x - ((n-1)*a + b), 0))
        def V(x):
            return sum(triangular(n,x) for n in n_values)
        x_pot = np.linspace(-5, 5, 1000)
        plt.plot(x_pot, V(x_pot))
        plt.title('Periodic Potential Function (PPF) for PLP')
        plt.xlabel('x')
        plt.ylabel('V(x)')

        plt.subplot(2, 2, 2)
        c_map=plt.cm.get_cmap('hsv', bands + 1)
        for i in range(bands):
            plt.plot(ka_1 / np.pi, E_1[i], '.', color=c_map(i), markersize=1)
        for x_coord in [-1, 1]:
            plt.axvline(x=x_coord, color='black', linestyle='--', linewidth=0.5)
        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Reduced Zone Scheme (RZS) for PLP')

        plt.subplot(2, 2, 3)
        for i in range(bands):
            plt.plot(ka_2 / np.pi, E_2[i], '.', color=c_map(i), markersize=1)
        for x_coord in [-6, 6]:
            plt.axvline(x=x_coord, color='black', linestyle='--', linewidth=0.5)
        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Periodic Zone Scheme (PZS) for PLP')

        plt.subplot(2, 2, 4)
        colors = ['k', 'k', 'k', 'k', 'k','k', 'k', 'k', 'k', 'k']
        for i in range(iter_ + 1):
            x_values = ka_3[i] / np.pi
            y_values = E_3[:, i]

            for band, color in zip(range(1, bands+1), colors):
                plt.plot(x_values, np.where(((x_values >= -band) & (x_values <= -band+1)) |
                                            ((x_values >= band-1) & (x_values <= band)),
                                            y_values[band-1], np.nan), color+'.', markersize=1)

        for x_coord in range(-bands, bands+1):
            plt.axvline(x=x_coord, color='brown', linestyle='--', linewidth=0.4)

        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Extended Zone Scheme (EZS) for PLP')
        plt.suptitle('Periodic Linear Potential (PLP)', color='c',fontsize=25)
        plt.figtext(0.5, 0.01, '\u00A9 Designed by Avoy Jana (MSc, IITD)', ha='center', va='top', fontsize=12, color='gray')
        plt.tight_layout()
        plt.show()

    # Interactive widget
    interact(calculate_and_plot,
            N_=IntSlider(min=5, max=100, step=5, value=default_N_, description='Basis limit', continuous_update=False),
            a=FloatSlider(min=0.1, max=10, step=0.1, value=default_a, description='a (A)', continuous_update=False),
            V0=FloatSlider(min=0, max=1000, step=10, value=default_V0, description='V0 (eV)', continuous_update=False),
            rho=FloatSlider(min=0, max=1, step=0.1, value=default_rho, description='b/a (rho)', continuous_update=False),
            iter_=IntSlider(min=10, max=1000, step=10, value=default_iter, description='# of iters', continuous_update=False),
            bands=IntSlider(min=1, max=10, value=default_bands, description='# of bands', continuous_update=False))
def PpCP():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import quad
    from ipywidgets import interact, IntSlider, FloatSlider

    # Default parameters
    default_N_ = 10
    N = 2 * default_N_ + 1
    default_a = 1.0
    default_rho = 0.5
    default_V0 = 100
    default_beta = 0.3
    default_iter = 200
    default_bands = 5
    h_pot = np.zeros((N, N))
    basis_arr = np.array([])

    def calculate_and_plot(N_, a, V0, beta, rho, iter_, bands):
        global default_N_, default_a, default_V0, default_beta, default_rho, h_pot, basis_arr
        default_N_ = N_
        default_rho = rho
        default_a = a
        default_V0 = V0
        default_beta = beta

        N = 2 * N_ + 1
        h_pot = np.zeros((N, N))

        basis_arr = np.array([np.sum([i * (-1) ** i for i in range(j + 1)]) for j in range(N)])
        seq_arr = np.array([i * (-1) ** i for i in range(N)])

        b = rho * a
        eta = 0.0103198*V0*(a** 2)
        fac = 10 ** (-10)
        def f(x):
            return 1-1/np.sqrt((x/(a*fac)-1/2-rho)**2+((beta*fac)/(a*fac))**2)
        def g(x,p,q):
            return np.exp(complex(0,1)*2*np.pi*(q-p)/(a*fac))*f(x)

        vpp=(1/(a*fac))*quad(f, b*fac, (a+b)*fac)[0]
        def vpq(p, q):
            return (1/(a*fac))*quad(g, b*fac, (a+b)*fac, args=(p, q))[0]

        for i in range(N):
            for j in range(N):
                if i == j:
                    h_pot[i, j] = eta * vpp
                else:
                    h_pot[i, j] = eta * vpq(i, j)

        E_1 = np.zeros((N, iter_ + 1))
        ka_1 = np.linspace(-np.pi, np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_1[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_1[:, i] = np.sort(np.real(E_eig))

        E_2 = np.zeros((N, iter_ + 1))
        ka_2 = np.linspace(-6*np.pi, 6*np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_2[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_2[:, i] = np.sort(np.real(E_eig))
        E_3 = np.zeros((N, iter_ + 1))
        ka_3 = np.linspace(-bands*np.pi, bands*np.pi, iter_ + 1)
        for i in range(iter_ + 1):
            h = np.copy(h_pot)
            for j in range(N):
                h[j, j] = h[j, j] + (2 * basis_arr[j] + ka_3[i] / np.pi) ** 2
            E_eig = np.linalg.eigvalsh(h)
            E_3[:, i] = np.sort(np.real(E_eig))

        plt.figure(figsize=(14, 14))
        plt.subplot(2, 2, 1)
        n_values = np.arange(-10, 11)
        def coulomb(n, x):
            return V0*(1 - a/np.sqrt((x-((2*n-1)/2*a+b))**2+beta**2))
        def V(x):
            return sum(coulomb(n,x) * (np.heaviside(x - ((n-1)*a+b), 0) - np.heaviside(x - (n*a+b), 0)) for n in n_values)
        x_pot = np.linspace(-5, 5, 1000)
        plt.plot(x_pot, V(x_pot))
        plt.title('Periodic Potential Function (PPF) for PpCP')
        plt.xlabel('x')
        plt.ylabel('V(x)')

        plt.subplot(2, 2, 2)
        c_map=plt.cm.get_cmap('hsv', bands + 1)
        for i in range(bands):
            plt.plot(ka_1 / np.pi, E_1[i], '.', color=c_map(i), markersize=1)
        for x_coord in [-1, 1]:
            plt.axvline(x=x_coord, color='black', linestyle='--', linewidth=0.5)
        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Reduced Zone Scheme (RZS) for PpCP')

        plt.subplot(2, 2, 3)
        for i in range(bands):
            plt.plot(ka_2 / np.pi, E_2[i], '.', color=c_map(i), markersize=1)
        for x_coord in [-6, 6]:
            plt.axvline(x=x_coord, color='black', linestyle='--', linewidth=0.5)
        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Periodic Zone Scheme (PZS) for PpCP')

        plt.subplot(2, 2, 4)
        colors = ['k', 'k', 'k', 'k', 'k','k', 'k', 'k', 'k', 'k']
        for i in range(iter_ + 1):
            x_values = ka_3[i] / np.pi
            y_values = E_3[:, i]

            for band, color in zip(range(1, bands+1), colors):
                plt.plot(x_values, np.where(((x_values >= -band) & (x_values <= -band+1)) |
                                            ((x_values >= band-1) & (x_values <= band)),
                                            y_values[band-1], np.nan), color+'.', markersize=1)

        for x_coord in range(-bands, bands+1):
            plt.axvline(x=x_coord, color='brown', linestyle='--', linewidth=0.4)

        plt.xlabel('$Ka/\pi\ \ $$\longrightarrow$')
        plt.ylabel('$E/E_1{(0)}\ \ $$\longrightarrow$')
        plt.title('Band Structure in Extended Zone Scheme (EZS) for PpCP')
        plt.suptitle('Periodic pseudo-Coulomb Potential (PpCP)', color='c',fontsize=25)
        plt.figtext(0.5, 0.01, '\u00A9 Designed by Avoy Jana (MSc, IITD)', ha='center', va='top', fontsize=12, color='gray')
        plt.tight_layout()
        plt.show()

    # Interactive widget
    interact(calculate_and_plot,
            N_=IntSlider(min=5, max=100, step=5, value=default_N_, description='Basis limit', continuous_update=False),
            a=FloatSlider(min=0.1, max=10, step=0.1, value=default_a, description='a (A)', continuous_update=False),
            V0=FloatSlider(min=0, max=1000, step=10, value=default_V0, description='V0 (eV)', continuous_update=False),
            beta=FloatSlider(min=0.1, max=5, step=0.1, value=default_beta, description='beta (A)', continuous_update=False),
            rho=FloatSlider(min=0, max=1, step=0.1, value=default_rho, description='b/a (rho)', continuous_update=False),
            iter_=IntSlider(min=10, max=1000, step=10, value=default_iter, description='# of iters', continuous_update=False),
            bands=IntSlider(min=1, max=10, value=default_bands, description='# of bands', continuous_update=False))
# Create a dropdown widget for choosing the code
code_selector = Dropdown(
    options={'Kroning Penney Potential (KPP)': KPP, 'Periodic Parabolic Potential (PPP)': PPP, 'Inverse PPP (IPPP)' : IPPP,
             'Periodic Gaussian Potential (PGP)' : PGP, 'Inverse PGP (IPGP)' : IPGP, 'Periodic Linear Potential (PLP)' : PLP,
             'Periodic pseudo-Coulomb Potential (PpCP)' : PpCP},
    value=KPP,
    description='Potential:'
)

# Define a function to run the selected code
def run_selected_code(selected_code):
    selected_code()

# Create an interactive widget
interact(
    run_selected_code,
    selected_code=code_selector
)
