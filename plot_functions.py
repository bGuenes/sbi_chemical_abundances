import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.stats import norm
from scipy.stats import multivariate_normal

##########################################################################################################
# --- Plot 1D Histogram ---

def plot_1d_hist(x1, x2, x_true, names, simulations):
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    plot_width = 0.3
    
    def plot_subplot(i, x, true, name):
        mean = np.mean(x)
        std = np.std(x)

        x_min = true - plot_width*true
        x_max = true + plot_width*true

        lin = np.linspace(x_min, x_max, 1000)
        p = norm.pdf(lin, mean, std)

        # fit of mean
        std_mu = std/np.sqrt(len(x)/simulations)
        mu = norm.pdf(lin, mean, std_mu)

        ax[i].hist(x, bins=500, density=True, alpha=0.6, color='g')
        ax[i].plot(lin, p, 'k', linewidth=2)
        ax[i].axvline(x=true, color='r', linestyle='dashed', linewidth=2)
        ax[i].set_title(fr"{name}: {true:.3f}"
                        "\n"
                        fr"$\mu$: {mean:.3f} $\sigma$: {std_mu:.3f}")
        ax[i].set_xlim(x_min, x_max)
        ax[i].plot(lin, p.max()*mu/mu.max())

    for i in [0,1]:
        if i == 0:
            x = x1
        else:
            x = x2

        plot_subplot(i, x, x_true[i], names[i])
    
    fig.suptitle(f'{int(len(x1)/simulations)} Stars', fontsize=30)
    plt.tight_layout()
    plt.show()


##########################################################################################################
# --- Plot 2D Histogram ---

def plot_2d_hist(x1, x2, x_true, N_stars):
    x_lim, y_lim = [-2.8,-1.8],[-3.4,-2.4]

    # --- Fit a 2D Gaussian to the data ---
    popt, pcov = multivariate_normal.fit(np.array([x1, x2]).T)
    x = np.linspace(x_lim[0], x_lim[1], 1000)
    y = np.linspace(y_lim[0], y_lim[1], 1000)

    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))

    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    rv = multivariate_normal(popt, pcov)
    perr = np.sqrt(np.diag(pcov))

    # --- Calculate the sigma levels ---
    levels =[]
    sigma = [3,2,1]
    for n in sigma:
        level = rv.pdf(popt+n * np.array([perr[0], perr[1]]))
        levels.append(level)

    # --- Plot the data ---
    plt.figure(figsize=(15,15))
    plt.hist2d(x1, x2, bins=500, range=[x_lim, y_lim])

    # draw contour lines on1,2,3 sigma
    CS = plt.contour(X, Y, rv.pdf(pos), levels=levels, colors='k', linestyles='dashed')
    text = plt.clabel(CS, inline=True, fontsize=10)
    c_labels = np.array(sigma)
    for i in range(len(text)):
        text[i].set(text=f'{c_labels[i]} $\\sigma$')

    plt.scatter(x_true[0], x_true[1], color='r', label='Ground Truth', s=10)
    plt.scatter(popt[0], popt[1], color='k', marker='x', label='Fit')

    plt.title(f'Posterior sampling of {N_stars} stars', fontsize=40)

    plt.xlabel(r'$\alpha_{\rm IMF}$', fontsize=20)
    plt.ylabel(r'$\log_{10} N_{\rm Ia}$', fontsize=20)
    plt.legend()
    plt.show();

##########################################################################################################
# --- Plot 2D Histogram with side plots ---

def plot_2d_hist_sides(x1, x2, x_true):
    x_lim, y_lim = [-2.8,-1.8],[-3.4,-2.4]

    # --- Fit a 2D Gaussian to the observed data ---
    popt, pcov = multivariate_normal.fit(np.array([x1, x2]).T)
    x = np.linspace(x_lim[0], x_lim[1], 1000)
    y = np.linspace(y_lim[0], y_lim[1], 1000)

    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))

    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    rv = multivariate_normal(popt, pcov)
    perr = np.sqrt(np.diag(pcov))# / np.sqrt(1000)

    # --- Calculate the sigma levels ---
    levels =[]
    sigma = [3,2,1]
    for n in sigma:
        level = rv.pdf(popt+n * np.array([perr[0], perr[1]]))
        levels.append(level)

    # --- Plot the data ---
    fig = plt.figure(1, figsize=(15,15))
    
    
    # Define the locations for the axes
    left, width = 0.12, 0.7
    bottom, height = 0.12, 0.7
    bottom_h = left_h = left+width+0.02
    
    # Set up the geometry of the three plots
    rect_temperature = [left, bottom, width, height] # dimensions of temp plot
    rect_histx = [left, bottom_h, width, 0.15] # dimensions of x-histogram
    rect_histy = [left_h, bottom, 0.15, height] # dimensions of y-histogram

    # Make the three plots
    axTemperature = plt.axes(rect_temperature) # temperature plot
    axHistx = plt.axes(rect_histx) # x histogram
    axHisty = plt.axes(rect_histy) # y histogram
    
    # Remove the inner axes numbers of the histograms
    nullfmt = NullFormatter()
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    axHistx.axis('off')
    axHisty.axis('off')


    # Plot the data
    axTemperature.hist2d(x1, x2, bins=500, range=[x_lim, y_lim])

    # draw contour lines on sigma levels
    CS = axTemperature.contour(X, Y, rv.pdf(pos), levels=levels, colors='k', linestyles='dashed')
    text = axTemperature.clabel(CS, inline=True, fontsize=10)
    c_labels = np.array(sigma)
    for i in range(len(c_labels)):
        text[i].set(text=f'{c_labels[i]} $\\sigma$')

    # labels
    label_gt = r'Ground Truth' + f"\n" + r"$\alpha_{\rm IMF} = $" + f'${round(x_true[0].item(), 2)}$' + f"\n" + r"$\log_{10} N_{\rm Ia} = $" + f'${round(x_true[1].item(), 2)}$'
    
    label_fit = r'Fit' + f"\n" + r"$\alpha_{\rm IMF} = $" + f'${round(popt[0].item(), 2)} \\pm {round(perr[0].item(),2)}$' + f"\n" + r"$\log_{10} N_{\rm Ia} = $" + f'${round(popt[1].item(), 2)} \\pm {round(perr[1].item(),2)}$'
    
    # plot the ground truth and the fit
    legend_true = axTemperature.scatter(x_true[0], x_true[1], color='r', label=label_gt, s=10)
    legend_fit = axTemperature.scatter(popt[0], popt[1], color='k', marker='x', label=label_fit)

    axTemperature.set_xlabel(r'$\alpha_{\rm IMF}$', fontsize=20)
    axTemperature.set_ylabel(r'$\log_{10} N_{\rm Ia}$', fontsize=20)
    

    # plot the histograms
    axHistx.hist(x1, bins=500, density=True, alpha=0.6, color='g')
    axHisty.hist(x2, bins=500, density=True, alpha=0.6, color='g', orientation='horizontal')

    axHistx.plot(x, norm.pdf(x, popt[0], perr[0]), 'k', linewidth=2)
    axHisty.plot(norm.pdf(y, popt[1], perr[1]), y, 'k', linewidth=2)

    axHistx.axvline(x=x_true[0], color='r', linestyle='dashed', linewidth=2)
    axHisty.axhline(y=x_true[1], color='r', linestyle='dashed', linewidth=2)

    axHistx.set_xlim(x_lim)
    axHisty.set_ylim(y_lim)

    #fig.text(0.85, 0.95, r'$\alpha_{\rm IMF} = $'+f'${x_true[0]}$', fontsize=20)
    #fig.text(0.85, 0.9, r'$\log_{10} N_{\rm Ia}$', fontsize=20)
    

    fig.legend(handles=[legend_true], fontsize=15, shadow=True, fancybox=True, loc=1, bbox_to_anchor=(0.99, 0.92))
    fig.legend(handles=[legend_fit], fontsize=15, shadow=True, fancybox=True, loc=1, bbox_to_anchor=(0.99, 0.99))
    plt.show()

##########################################################################################################
# --- N-Star parameter plot ---

def n_stars_plot(x1, x2, x_true, no_stars= np.array([1, 10, 100, 500, 1000]), simulations=1000):
    fit = []
    err = []

    # --- Fit a 2D Gaussian to the data ---
    for n in no_stars:
        samples = int(n*simulations)
        popt, pcov = multivariate_normal.fit(np.array([x1[0:samples], x2[0:samples]]).T)

        mu1 = popt[0]
        mu2 = popt[1]
        sigma1 = np.sqrt(pcov[0,0]) / np.sqrt(n)
        sigma2 = np.sqrt(pcov[1,1]) / np.sqrt(n)

        fit.append([mu1, mu2])
        err.append([sigma1, sigma2])
        

    fit = np.array(fit)
    err = np.array(err)

    # --- Plot the data ---
    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(26,6))

    # Plot alpha_IMF
    ax[0].plot(no_stars, fit[:,0], color="b", label="Fit")
    ax[0].fill_between(no_stars, fit[:,0]-err[:,0], fit[:,0]+err[:,0], alpha=0.3,color="b", label=r"1 & 2 $\sigma$")
    ax[0].fill_between(no_stars, fit[:,0]-2*err[:,0], fit[:,0]+2*err[:,0], alpha=0.2,color="b")

    ax[0].axhline(x_true[0], color='k', linestyle=':', linewidth=2, label='Ground Truth')

    ax[0].set_xlabel(r'$N_{\rm stars}$', fontsize=20)
    ax[0].set_ylabel(r'$\alpha_{\rm IMF}$', fontsize=20)
    ax[0].set_ylim([-2.6,-2.0])
    ax[0].set_xlim([0,1000])
    ax[0].legend(fontsize=15, fancybox=True, shadow=True)

    # Plot log10_N_Ia
    ax[1].plot(no_stars, fit[:,1], color="b", label="Fit")
    ax[1].fill_between(no_stars, fit[:,1]-err[:,1], fit[:,1]+err[:,1], alpha=0.3,color="b", label=r"1 & 2 $\sigma$")
    ax[1].fill_between(no_stars, fit[:,1]-2*err[:,1], fit[:,1]+2*err[:,1], alpha=0.2,color="b")

    ax[1].axhline(x_true[1], color='k', linestyle=':', linewidth=2, label='Ground Truth')

    ax[1].set_xlabel(r'$N_{\rm stars}$', fontsize=20)
    ax[1].set_ylabel(r'$\log_{10} N_{\rm Ia}$', fontsize=20)
    ax[1].set_ylim([-3.1,-2.6])
    ax[1].set_xlim([0,1000])
    #ax[1].legend(fontsize=15, fancybox=True, shadow=True)

    plt.show()

##########################################################################################################