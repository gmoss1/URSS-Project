# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:16:56 2019

@author: Guy
"""


    
##############################################################################
#Pipeline used to obtain parameters from KASOC-concatenated timeseries data  #
# for sun-like stars. The methodology follows that described by Hekker et al.#
# (2009) in the following paper: https://arxiv.org/pdf/0911.2612.pdf         #
# First, the Lomb-Scargle periodogram is computed from the timeseries. The   #
#background is then fitted using two Harvey profiles*, in addition to        #
#instrument noise. Subtracting the background from the overall periodogram,  #
#the peak separation delta-nu is found, before the data is heavily smoothed  #
#to an excess-power hump, which is then fitted. The fit parameters are then  #
#used, along with the background parameters, as the initial guess for an MCMC#
#algorithm which fits the entire periodogram simultaneously. The maxmimum    #
#mode amplitude can then be computed using the best MCMC result.             #
#The code computes the sph (photospheric activity index) and maximum mode    #
#amplitude for all the stars stored in the folder containing .fits files     #
#and summarises (and saves) the results to a table using python's pickle.    #
#The code takes roughly 10-15 minutes per star to run.                       #
############################################################################## 


##############################################################################
#*When running the code, the main() function specifies whether the background#
# is fitted with the activity component (True) or without it (False). Without#
# activity the code runs slightly faster. There is no significant difference #
#in the final results produced by the code using the two different methods.  #
##############################################################################


"""
Importing the relevant modules.
"""


import numpy as np
from astropy.io import fits
from astropy.timeseries import LombScargle
import scipy.signal as sig
import time
import emcee
from scipy.optimize import curve_fit
from pandas import Series
#import matplotlib.pyplot as plt
import os
import pickle
from tabulate import tabulate



"""
Start Time to be used when printing time elapsed.
"""
start_time = time.time()

"""
Constants for window calculations (finding oscillation range in Lomb-Scargle
periodogram)
"""
nu_max_sol = 3100
width_sol = 2000
nu_central_init = 6200.0/21.0


"""
Directories for saving.
fits_directory - folder in which the .fits files containing timeseries data
                 for each star is saved.
text_directory - folder in which output files are saved so they can be accessed
                  without running the entire code.
figures_directory - folder in which figures produced by the graph are saved.

These will have to be modified for the machine you are using.
"""
fits_directory = r"C:\Users\Guy\.spyder-py3\URSS\fits_files\\"
text_directory = r"C:\Users\Guy\.spyder-py3\URSS\output_files\\"
figures_directory = r"C:\Users\Guy\.spyder-py3\URSS\figures\\"






"""
This function, used throughout the code, finds the index of the element in an
array closest to a specified value.
"""

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx




"""
This function takes the timeseries data, computes the lomb-scargle periodogram, 
and then searches for the presence of oscillations using a simplified version of
the methodology described in the Octave pipeline (see top). It also returns the 
approximate range in which the oscillations are found so that it can be discarded 
when computing the background profile, and the sph value of the timeseries
"""

def finding_freq_range(kic_number,fits_filename):
    global nu_central_init
    
    """
    Obtains a section of an array given the centre of the section and its width, in
    microHerz.
    """
    def window(freq,power,center,width):
        signal_freq_range = freq[(freq>center-width/2)&(freq<center+width/2)].copy()
        signal_freq_range -= signal_freq_range[0]
        signal_power_wanted = power[(freq>center-width/2)&(freq<center+width/2)].copy()
        signal_power_wanted *= 1e6
        signal_power_wanted -= np.mean(signal_power_wanted)
        signal_freq_range *= 1e-6
        #spacing = signal_freq_range[1] - signal_freq_range[0]
        return signal_freq_range,signal_power_wanted
    
    """
    Computes the power spectrum of the given window in frequency and power.
    Only considers frequencies below 300 microHerz given where we expect to find
    peaks.
    """
    def psps(freq,power):
        ft_time,ft_power = LombScargle(freq,power).autopower(nyquist_factor=1)
        ft_freq = (1.0e6)/(ft_time)
        ft_power = ft_power[ft_freq<300]
        ft_freq = ft_freq[ft_freq<300]
        #print(ft_freq.size)
        #print(ft_freq)
        return ft_freq,ft_power
    
    """
    Given the formula delta_nu ~ 0.263*nu_max**(0.772) for the large frequency 
    separation delta-nu given the frequency at which the oscillations are at a 
    maximum amplitude, nu-max, compute the minimum and maximum value, with 30% 
    tolerance, for where we would see peaks in the PSPS if the window we are 
    observing contains oscillations.
    """
    def delta_nu_range(center):
        delta_nu = 0.263*(center**0.772)
        return (0.7*delta_nu),(1.30*delta_nu)
    
    """
    Checks for the presence of peaks at the theoretically-predicted values of 
    1/6, 1/4, and 1/2 of the expected delta_nu values.
    """
    def oscillation_detection(freq,peaks,nu_min,nu_max):
        if any((freq[peaks]*6 >nu_min)&(freq[peaks]*6<nu_max)) and\
            any((freq[peaks]*4 >nu_min)&(freq[peaks]*4<nu_max)) and\
            any((freq[peaks]*2 >nu_min)&(freq[peaks]*2<nu_max)):
                return('Oscillations Detected!')
        else:
            return('No Oscillations')
    
    """
    Using the initial window defined by constants defined at the top of the 
    document, this function calculates the PSPS to check for the presence of
    oscillations in the window, before updating the parameters of the window to
    iterate through the next one. If oscillations are found, the oscillation range is
    updated. If no oscillations are found, this function returns [0,0] for the 
    oscillation range, which is then interpreted as "no oscillations", which is
    considered by the code an error, stopping the rest of the code from running.
    """
    def window_iterating(freq,power):
        global nu_max_sol,width_sol,nu_central_init
        nu_central = nu_central_init
        width = 2000*nu_central_init/3100
        freq_windows = []
        power_windows = []
        detections = []
        oscillation_freq_range= np.array([0.0,0.0])
        while True:
            if (nu_central + width/2) > np.amax(freq):
                #freq_windows.append(window(freq,power,nu_central,(np.amax(freq)-nu_central)))
                freq_windows.append(freq[freq>(nu_central-width/2)])
                power_windows.append(power[freq>(nu_central-width/2)])
                signal_freq_range,signal_power_wanted = window(freq[(freq>(nu_central-width/2))],\
                                                                    power[(freq>(nu_central-width/2))],\
                                                                    nu_central,width)
                ft_freq,ft_power = psps(signal_freq_range,signal_power_wanted)
                
                """
                Get rid of frequency values below 1 microHerz as it is dominated by noise
                and is irrelevant to the search - not interested in any peaks there.
                """
                ft_power = ft_power[ft_freq>1]
                ft_freq = ft_freq[ft_freq>1]
                
                """
                For increased accuracy, we first check the highest peak in the 
                PSPS, to see whether it is within tolerance of the expected delta-nu/2 
                peak. If not, then we do not bother checking for other peaks.
                """ 
                nu_min,nu_max = delta_nu_range(nu_central)
                idx = find_nearest(ft_power,np.amax(ft_power))
                nu_peak = ft_freq[idx]
                if (2*nu_peak > nu_min) and (2*nu_peak < nu_max):
                    peaks, _ = sig.find_peaks(ft_power[(ft_freq>(0.1*nu_min))&(ft_freq<(10*nu_max))],\
                                                       #np.amax(ft_power[(ft_freq>(0.1*nu_min))&(ft_freq<(10*nu_max))])/10,\
                                                       prominence=np.amax(ft_power/20))
                    """
                    This is another addition to improve accuracy. If there is a large
                    number of peaks, then the code has detected noise rather than a signal.
                    In this case, we discard the detection. The choice of 30 was obtained 
                    through some trial and error, and has no significance theoretically.
                    """
                    if peaks.size <30:
                        detection = oscillation_detection(ft_freq,peaks,nu_min,nu_max)
                        detections.append(detection)
                    else:
                        detection = 'No Oscillations'
                        detections.append(detection)
                
                else:
                    detection = 'No Oscillations'
                    detections.append(detection)
                if detection == 'Oscillations Detected!':
                    if np.abs(oscillation_freq_range[0]) < 0.1:
                        oscillation_freq_range[0] = nu_central-width/4
                        oscillation_freq_range[1] = nu_central + width/4
                    else:
                        if (nu_central-width/16)>oscillation_freq_range[1]:
                            oscillation_freq_range[1] = nu_central - width/16
                return freq_windows,power_windows,detections,oscillation_freq_range
            
                """
                This is a repetition of the code above, in the case that we are not on the
                last window, and so we update the values at the end to the next window as
                opposed to returning a result.
                """
            else:
                #freq_windows.append(window(freq,power,nu_central,width))
                freq_windows.append(freq[(freq>(nu_central-width/2))&(freq<(nu_central+width/2))])
                power_windows.append(power[(freq>(nu_central-width/2))&(freq<(nu_central+width/2))])
                signal_freq_range,signal_power_wanted = window(freq[(freq>(nu_central-width/2))&(freq<(nu_central+width/2))],\
                                                                    power[(freq>(nu_central-width/2))&(freq<(nu_central+width/2))],\
                                                                    nu_central,width)
                ft_freq,ft_power = psps(signal_freq_range,signal_power_wanted)
                ft_power = ft_power[ft_freq>1]
                ft_freq = ft_freq[ft_freq>1]
                nu_min,nu_max = delta_nu_range(nu_central)
                idx = find_nearest(ft_power,np.amax(ft_power))
                nu_peak = ft_freq[idx]
                if (2*nu_peak > nu_min) and (2*nu_peak < nu_max):
                    peaks, _ = sig.find_peaks(ft_power[(ft_freq>(0.1*nu_min))&(ft_freq<(10*nu_max))],\
                                                       #np.amax(ft_power[(ft_freq>(0.1*nu_min))&(ft_freq<(10*nu_max))])/10,\
                                                       prominence=np.amax(ft_power/20))
                    if peaks.size <30:
                        detection = oscillation_detection(ft_freq,peaks,nu_min,nu_max)
                        detections.append(detection)
                    else:
                        detection = 'No Oscillations'
                        detections.append(detection)
                
                else:
                    detection = 'No Oscillations'
                    detections.append(detection)
                if detection == 'Oscillations Detected!':
                    if np.abs(oscillation_freq_range[0]) < 0.1:
                        oscillation_freq_range[0] = nu_central-width/4
                        oscillation_freq_range[1] = nu_central + width/4
                    else:
                        if (nu_central - width/16)>oscillation_freq_range[1]:
                            oscillation_freq_range[1] = nu_central - width/16
                    
                nu_central += width/4
                width = 2000*nu_central/3100
                
                
    """
    open fits file
    """
    print('Reading Data...')
    hdul = fits.open(fits_directory+fits_filename)
    
    
    """
    read data info (if necessary)
    """
    #hdul.info()
    
    
    """
    Information about what data is available, units, etc.
    """
    #hdr = hdul[1].header
    #print(repr(hdr))
    
    
    """
    extract relevant data from file
    """
    tme = hdul[1].data['TIME']
    tme *= (3600*24)
    flux = hdul[1].data['FLUX']
    flux_err = hdul[1].data['FLUX_ERR']
    
    """
    Data has some NaN values that need to be removed.
    The frequency-flux-flux_err data is combined and then frequency-
    flux-flux_err triplets where at least one of the values is a NaN
    are removed. The time,flux,flux-error arrays are then 
    updated.
    """
    combined = np.array([tme,flux,flux_err])
    combined = combined.transpose()
    combined = combined[~np.isnan(combined).any(1)]
    combined = combined.transpose()
    
    tme = combined[0]
    flux = combined[1]
    flux_err = combined[2]
    
    
    """
    close the file
    """
    hdul.close()
    print('Done!')
    print("--- %s seconds ---" % (time.time() - start_time))

    """
    If data needs to be shortened for faster (but less precise) computation
    """
    #flux = flux[0::10]
    #time = time[0::10]
    
    
    """
    Centre the data for both time and flux
    """
    tme = tme - np.amin(tme)
    flux_mean = np.mean(flux)
    flux = flux - flux_mean
    
    
    sph = np.std(flux)
    """
    Compute the Lomb-Scargle periodogram for the data.
    The nyquist_factor is set to 1 so that the algorithm does not-
    look for irrelevant frequencies in the data.
    """
    print('Computing Lomb-Scargle Periodogram...')
    freq,power = LombScargle(tme,flux,flux_err).autopower(nyquist_factor=1)
    
    
    """
    Convert the power units in the resulting periodogram to ppm/microHz
    """
    ms_power = 1/tme.size *np.sum(flux**2)
    power = power*ms_power/(np.sum(power)*(freq[1]-freq[0])*1e6)

    """
    Rescale the frequency data to show data in microH\
    """
    freq *= 10**6
    print('Done!')
    print("--- %s seconds ---" % (time.time() - start_time))


    
    
    
    
    """
    Run the window_iteration function defined above to find the oscillation range.
    A few other parameters are returned to be printed as a sanity check to make sure
    nothing unexpected occured.
    """
    print('Searching for Oscillations...')
    freq_windows,power_windows,detections,oscillation_freq_range = window_iterating(freq,power)
    print('Done!')
    print("--- %s seconds ---" % (time.time() - start_time))

    
    print('Detections: ')
    print(detections)
    print('Number of windows (two numbers must match): ')
    print(len(freq_windows))
    print(len(power_windows))
    print('Oscillation Frequency Range: ')
    print(oscillation_freq_range)
    
    """
    Code that raises an error if the oscillation range was found to be [0,0]
    so that star will be skipped.
    """
    wanted_freq_range,wanted_power_range = window(freq,power,np.mean(oscillation_freq_range),oscillation_freq_range[1]-oscillation_freq_range[0])
    ft_freq,ft_power = psps(wanted_freq_range,wanted_power_range)
    peaks, _ = sig.find_peaks(ft_power,np.amax(ft_power)/10,prominence=0.004)
    nu_min,nu_max = delta_nu_range(np.mean(wanted_freq_range))
    
    
    
    """
    Return frequency and power arrays along with the oscillation frequency range
    and the sph.
    """
    return freq,power,oscillation_freq_range,sph
   
        
"""
Smooth the periodogram using a boxcar filter of size 100 data points.
This increases the computation speed and precision of the fitting functions
in the code.
"""
        
def smoothing(kic_number,freq_input,power_input):
    
    freq = freq_input.copy()
    power = power_input.copy()
    

    
    """
    Smoothing uses the pandas.Series.rolling function.
    """
    def boxcar(power_array):
        window_size = 100
        print('Window size: ' , window_size)
        series = Series(power_array,index = freq)
        rolling = series.rolling(window = window_size,center = True,min_periods = 100)
        rolling_mean = rolling.mean()
        rolling_mean = rolling_mean.to_numpy(dtype = float)
        rolling_std = rolling.std()
        rolling_std = rolling_std.to_numpy(dtype = float)
        return rolling_mean,rolling_std
    
    """
    The smoothing function produces some NaN values at both ends of the data,
    so these are removed.
    """
    boxcar_power,boxcar_std = boxcar(power)
    combined = np.array([freq,boxcar_power,boxcar_std,power])
    combined = combined.transpose()
    combined = combined[~np.isnan(combined).any(1)]
    combined = combined.transpose()
    
    freq = combined[0]
    boxcar_power = combined[1]
    boxcar_std = combined[2]
    power = combined[3]
    
    """
    Return the smoothed periodogram along with the smoothing error.
    """
    return freq,boxcar_power,boxcar_std

    
    
"""
This function computes the background fit as described at the top of the document.
There are two types of background fits, both similar. The 'Simple' one is the one used
by the Octave pipeline referenced above. The second model is Model F as described by 
Kallinger et al. in https://arxiv.org/pdf/1408.0817.pdf
Note that the Kallinger et al. model has not been tested to the same extent and might 
produce unexpected results.
"""
def total_background(kic_number,activity,freq_input,power_input,power_err_input,oscillation_range):
    freq = freq_input.copy()
    power = power_input.copy()
    power_err = power_err_input.copy()
    
    
    """
    This is a part of the Kallinger model, a modifier to the two Harvey profiles.
    """
    def eta_correction(freq):
        nu_nyq = np.amax(freq)
        x = ((np.pi/2)*(freq/nu_nyq))
        return (np.sin(x)/x)**2 
    
    
    
    """
    Background Functions to be passed for curve_fit.
    """
    
    """
    Simple backgound model, the functions in order are:
    bg_total - The total background power - bg_total is defined later
    bg_gran - The granulation profile
    bg_act - The activity profile
    bg_offset - the constant instrumental noise
    """
    
    
    
    def bg_gran(freq,p_gran,tau_gran,a):
        x = p_gran/ ((1+(tau_gran*freq)**a))
        return x
    
    def bg_act(freq,p_act,tau_act):
        x= p_act/(1+(tau_act*freq)**2)
        return x
    
    def bg_offset(freq,b):
        x = np.full(freq.size,b)
        return x
    
    
    """
    Complex background model - the functions are the same as for the simple model.
    """
    
    """
    def bg_gran(freq,p_gran,tau_gran,a_gran):
        x = eta_correction(freq)*(p_gran*tau_gran/ ((1+(tau_gran*freq)**a_gran)))
        return x
    
    def bg_act(freq,p_act,tau_act,a_act):
        x= eta_correction(freq)*(p_act*tau_act/(1+(tau_act*freq)**a_act))
        return x
    
    def bg_offset(freq,b):
        x = np.full(freq.size,b)
        return x
    """
    
    
    
    """
    Based on whether activity is being used as part of the background, define
    the total background function, along with the appropriate ranges of the 
    Lomb-Scargle periodogram for which to fit the background. Note that if 
    activity is being ignored, we also discard the periodogram values where
    frequency is smaller than 100 microHz, as that is where activity is the 
    strongest. 
    Define the initial parameter guesses to be passed to curve_fit.
    """
    
    
    if activity == True:
        
        
        def bg_total(freq,p_gran,tau_gran,a,p_act,tau_act,b):
            x=  p_gran/((1+(tau_gran*freq)**a)) + p_act/(1+(tau_act*freq)**2) + b
            return x
        
        
        """
        def bg_total(freq,p_gran,tau_gran,a_gran,p_act,tau_act,b):
            x=  eta_correction(freq)*(p_gran*tau_gran/((1+(tau_gran*freq)**a_gran)) + p_act*tau_act/(1+(tau_act*freq)**2)) + b
        return x
        """
        
        power_exc_signal = np.delete(power,np.where((oscillation_range[0]<freq) & (freq <oscillation_range[1])))
        power_err_exc_signal = np.delete(power_err,np.where((oscillation_range[0]<freq) & (freq <oscillation_range[1])))
        freq_exc_signal = np.delete(freq,np.where((oscillation_range[0]<freq) & (freq <oscillation_range[1])))
        high_freq_lim = power_exc_signal[np.where(freq_exc_signal>(1.0*oscillation_range[1]))]
        
        """
        Complex background model parameter guesses
        """
    
        """
        p_gran_guess = np.max(power_exc_signal)*1e2
        tau_gran_guess = 1e-3
        tau_act_guess = 1e-1
        p_act_guess = np.max(power_exc_signal)*1e1
        b_guess = np.mean(high_freq_lim)
        a_gran_guess = 2
        """
        
        
        """
        Simple background model parameter guesses
        """
        
        
        p_gran_guess = np.max(power_exc_signal)
        tau_gran_guess = 1e-3
        a_gran_guess = 2
        p_act_guess = np.max(power_exc_signal)
        tau_act_guess = 1e-1
        b_guess = np.mean(high_freq_lim)
        
        x0= np.array([p_gran_guess,tau_gran_guess,a_gran_guess,p_act_guess,tau_act_guess,b_guess])

    else:
        
        
        def bg_total(freq,p_gran,tau_gran,a,b):
            x=  p_gran/((1+(tau_gran*freq)**a))  + b
            return x
        
        """
        def bg_total(freq,p_gran,tau_gran,a_gran,b):
            x=  eta_correction(freq)*(p_gran*tau_gran/((1+(tau_gran*freq)**a_gran))) + b
        return x
        """
        
        
        power_exc_signal = np.delete(power,np.where((oscillation_range[0]<freq) & (freq <oscillation_range[1])))
        power_exc_signal = np.delete(power_exc_signal,np.where(freq<100))
        power_err_exc_signal = np.delete(power_err,np.where((oscillation_range[0]<freq) & (freq <oscillation_range[1])))
        power_err_exc_signal = np.delete(power_err_exc_signal,np.where(freq<100))
        freq_exc_signal = np.delete(freq,np.where((oscillation_range[0]<freq) & (freq <oscillation_range[1])))
        freq_exc_signal = np.delete(freq_exc_signal,np.where(freq_exc_signal<100))
        high_freq_lim = power_exc_signal[np.where(freq_exc_signal>(1.0*oscillation_range[1]))]
        
        power = np.delete(power,np.where(freq<100))
        power_err = np.delete(power_err,np.where(freq<100))
        freq = np.delete(freq,np.where(freq<100))    
        
        """
        In cases where the activity is significant at low frequencies but you still wish to 
        ignore it, remove a larger section of the periodogram at the lower frequency end.
        """
        
        
        if np.max(power_exc_signal) > 500:
            power_exc_signal = np.delete(power_exc_signal,np.where(freq<300))
            power_err_exc_signal = np.delete(power_err_exc_signal,np.where(freq<300))
            freq_exc_signal = np.delete(freq_exc_signal, np.where(freq<300))
            power = np.delete(power,np.where(freq<300))
            power_err = np.delete(power_err,np.where(freq<300))
            freq = np.delete(freq,np.where(freq<300))
            
            
        """
        Complex background model parameter guesses
        """
    
        """
        p_gran_guess = np.max(power_exc_signal)*1e2
        tau_gran_guess = 1e-3
        b_guess = np.mean(high_freq_lim)
        a_gran_guess = 2
        """
        
        
        """
        Simple background model parameter guesses
        """
        
        
        p_gran_guess = np.max(power_exc_signal)
        tau_gran_guess = 1e-3
        a_gran_guess = 2
        b_guess = np.mean(high_freq_lim)
        
        x0 = np.array([p_gran_guess,tau_gran_guess,a_gran_guess,b_guess])
    
    
    
    
    
    
    """
    Apply the least_squares fitting function, curve_fit (from scipy.optimize library)
    Note that the error "sigma" is used as otherwise the fitting process is heavily
    skewed to fit the smaller number of points at much higher power values at lower frequencies
    """
    
    noise_sigma = power_err_exc_signal

    print('Initial Parameter Guess: ' , x0)
    print('Optimizing...')
    popt,pcov = curve_fit(bg_total,freq_exc_signal,power_exc_signal,p0=x0,\
                          sigma = noise_sigma , bounds=(0,np.inf))
    print('Done!')
    print("--- %s seconds ---" % (time.time() - start_time))

    params = popt
    print('Optimized Background Parameters: ' ,params)
    
    
    
    """
    Compute each of the profiles using the optimized fit parameters.
    """
    
    bg_total_power = bg_total(freq,*(popt[i] for i in range(0,len(popt))))
    bg_gran_power = bg_gran(freq,*(popt[i] for i in range(0,3)))
    if activity == True:
        bg_act_power = bg_act(freq,*(popt[i] for i in range(3,len(popt)-1)))
    bg_offset_power = bg_offset(freq,popt[len(popt)-1])
    
    
    """
    Return the frequency,power, and background power arrays.
    While all components are returned, only the total bg_total_power will be used
    for the rest of the code.
    """
    
    if activity == True:
        return freq,power,bg_gran_power,bg_act_power,bg_offset_power,bg_total_power
    else:
        return freq,power,bg_gran_power,bg_offset_power,bg_total_power

    


"""
This function finds the large frequency separation, delta-nu. 
Even if you are not interested in this parameter, this value is important for the rest
of the code, so it cannot be skipped.
"""
def find_delta_nu(kic_number,freq_input,power_input,bg_power_input,oscillation_range):
    freq = freq_input.copy()
    power = power_input.copy()
    bg_power = bg_power_input.copy()
    
    """
    Take away the total background from the lomb-scargle power to obtain the 
    residual power, where we expect to only see the oscillations standing out.
    """
    signal_power = power - bg_power
    
    """
    Straight line function - will be fitted later on.
    """
    
    
    def straight_line(x,m,c):
        return m*x + c
    
    
    """
    Smoothing functoin - this accentuates the peaks further, and so makes them 
    easier to find reliably and tell them apart from peaks due to random noise.
    """
    def smooth(freq_array,power_array,window_size):
        print('Window size: ' , window_size)
        series = Series(power_array,index = freq_array)
        rolling = series.rolling(window = window_size, center = True,min_periods = 100)
        rolling_mean = rolling.mean()
        rolling_mean = rolling_mean.to_numpy(dtype = float)
        combined = np.array([freq_array,rolling_mean])
        combined = combined.transpose()
        combined = combined[~np.isnan(combined).any(1)]
        combined = combined.transpose()
        return combined[0],combined[1]  
    
    
    """
    Guess the initial parameters of the straight line
    """
    delta_nu_guess = 100.0
    offset_guess = 10.0
    
    
    """
    Obtain the window in which the oscillations are, and smooth using a boxcar filter
    of size 2000 data points. The value of 2000 is again found using trial and error
    for the kasoc-fits timeseries data used, and might have to be modified for other
    timeseries data.
    """
    signal_freq_range = freq[np.where((freq>oscillation_range[0])&(freq<oscillation_range[1]))]
    print(signal_freq_range.size)
    signal_power_wanted = signal_power[np.where((freq>oscillation_range[0])&(freq<oscillation_range[1]))]
    signal_freq_range,signal_power_wanted = smooth(signal_freq_range,signal_power_wanted,2000)

    
    """
    Obtain an underestimate for delta-nu and half it to use as a minimum distance 
    between consecutive peaks, to reduce the probability of peaks being identified 
    where there is only random noise.
    """
    idx = find_nearest(signal_power_wanted,np.amax(signal_power_wanted))
    numax = signal_freq_range[idx]
    print('NuMax: ' , numax)
    delta_nu_min = 0.263*(numax**0.772)*0.7
    print('DeltaNuMin: ' , delta_nu_min)
    distance_min = idx - find_nearest(signal_freq_range,numax - delta_nu_min/2)
    
    """
    Find the peaks and separate into the (2) modes that are dominant.
    The minimum height for the peaks is specified as a fraction of the maximum power
    in the array. This is repeated for several different factors.
    The factors that produce reasonable results are then taken, and their average is 
    calculated. This reduces the chance that anomalies in the data lead to an incorrect
    value for delta-nu.
    """
    print('Searching for Peaks...')
    delta_nus = np.array([])
    i_values = np.array([])
    for i in range(31):
        peaks,_ = sig.find_peaks(signal_power_wanted,np.amax(signal_power_wanted)/(2.0+0.1*i),distance=distance_min)
        l1_freqs = signal_freq_range[peaks[0::2]]
        l0_freqs = signal_freq_range[peaks[1::2]]
        nl0 = np.arange(l0_freqs.size)
        nl1 = np.arange(l1_freqs.size)
        poptl0,pcovl0 = curve_fit(straight_line,nl0,l0_freqs,p0=(delta_nu_guess,offset_guess))
        poptl1,pcovl1 = curve_fit(straight_line,nl1,l1_freqs,p0=(delta_nu_guess,offset_guess))
        delta_nu_l0 = poptl0[0]
        delta_nu_l1 = poptl1[0]
        if np.abs(delta_nu_l0 - delta_nu_l1) <2.0:
            delta_nus = np.append(delta_nus,(delta_nu_l0+delta_nu_l1)/2)
            i_values = np.append(i_values,i)
    print('Done!')
    print("--- %s seconds ---" % (time.time() - start_time))
    print('Delta Nu Array:')
    print(delta_nus)
    print('Factor Values:')
    print(2.0+0.1*i_values)
    
    while delta_nus.size >8:
        outlier_idx = np.abs(delta_nus - np.mean(delta_nus)).argmax()
        delta_nus = np.delete(delta_nus,outlier_idx)
        i_values = np.delete(i_values,outlier_idx)
        
        
    """
    Print the best value of delta-nu found and save the plot as a sanity check.
    """
    delta_nu = np.mean(delta_nus)
    print('Delta Nu: ')
    print(delta_nu)
    chosen_i = i_values[np.abs(delta_nus-np.mean(delta_nus)).argmin()]
    print('Best Factor: ')
    print(2.0+0.1*chosen_i)
    
    print('Plotting Best Result...')
    peaks, _ = sig.find_peaks(signal_power_wanted,np.amax(signal_power_wanted)/(2.0+0.1*chosen_i),distance = distance_min)
    print('Peaks Found at following frequencies:')
    print(signal_freq_range[peaks])
    l1_freqs = signal_freq_range[peaks[0::2]]
    l0_freqs = signal_freq_range[peaks[1::2]]
   
    
    
    
    """
    Define the arrays on which we plot the frequencies
    """
    nl0 = np.arange(l0_freqs.size)
    nl1 = np.arange(l1_freqs.size)
    
    
    
    
    """
    Run the fitting algorithm for both modes. Delta-nu will be the gradient of the 
    straight lines produced by the fit.
    """
    print('Finding Gradient...')
    poptl0,pcovl0 = curve_fit(straight_line,nl0,l0_freqs,p0=(delta_nu_guess,offset_guess))
    print('Done!')
    print("--- %s seconds ---" % (time.time() - start_time))

    delta_nu_l0 = poptl0[0]
    offset_l0 = poptl0[1]
    poptl1,pcovl1 = curve_fit(straight_line,nl1,l1_freqs,p0=(delta_nu_guess,offset_guess))
    delta_nu_l1 = poptl1[0]
    offset_l1 = poptl1[1]

    
    """
    Return the value of delta-nu for other parts of the code to use.
    """
    return delta_nu 
    
   


"""
This function uses least-squares again to fit the peak after the background has
been subtracted. While the power excess hump is usually fitted using a gaussian,
I have found that at times a lorentzian might produce a better fit and so both 
are fitted here, and plotted so that they can be compared.
By deafult, the MCMC algortihm later on uses a gaussian fit for the power excess
hump, but only a few tedious changes are required to change the fit it uses to 
a lorentzian, and no major changes to the code are necessary.
"""
def peak_fitting(kic_number,freq_input,power_input,background_power,oscillation_range,delta_nu):
    
    
    """
    The Gaussian and Lorentzian functions.
    """
    def gaussian(x,scale,mu,sd):
        return scale*(1/np.sqrt(2*np.pi*sd**2))*np.exp(-0.5*((x-mu)/sd)**2)
    
    
    def lorentzian(x,scale,mu,width):
        return (scale/width)/(1 + (2*(x-mu)/width)**2)

    freq = freq_input.copy()
    power = power_input.copy()

    """
    Compute the "signal power" as the difference between the total power and the
    background power. Then find the maximum of that array and the point at which
    it is achieved.
    """ 
    signal_power = power-background_power
    
    """
    Work out the relevant frequency range of the oscillations we are looking for.
    Also work out the relevant width, as given by the Octave pipeline, of the
    boxcar filter for binning purposes.
    """
    
    idx_top = find_nearest(freq, oscillation_range[1])
    idx_bottom = find_nearest(freq, oscillation_range[0])
    wanted_freq_range = freq[idx_bottom:idx_top]
    wanted_power_range = signal_power[idx_bottom:idx_top]
    idx = find_nearest(wanted_power_range,np.amax(wanted_power_range))
    nu_max = wanted_freq_range[idx]
    print('vmax:' , nu_max)
    max_power = np.amax(wanted_power_range)
    print('max power:' , max_power)
    
    print(delta_nu)
    window_size = find_nearest(freq,nu_max + 1.0*delta_nu) - find_nearest(freq,nu_max - 1.0*delta_nu)
    print(window_size)
    
    """
    Compute and print the signal to noise ratio (SNR). If it is low when the 
    oscillations are quite prominenet, something has gone wrong.
    """
    print('Computing Signal to Noise (SNR) Ratio...')
    p_tot = np.sum(signal_power[idx_bottom:idx_top])
    b_tot = np.sum(background_power[idx_bottom:idx_top])
    snr_tot = p_tot/b_tot+1
    print('Done!')
    print('SNR Total:' , snr_tot)
    
    """
    Use a moving average to bin the power data. Remove any NaN values that are
    produced by the moving average calculator (on the edges).
    """
    
    print('Applying heavy smoothing...')
    series = Series(wanted_power_range,index = wanted_freq_range)
    print(window_size)
    rolling = series.rolling(window = window_size,center = True,min_periods=100)
    rolling_mean = rolling.mean()
    rolling_mean = rolling_mean.to_numpy(dtype = float)
    
    combined = np.array([wanted_freq_range,rolling_mean])
    combined = combined.transpose()
    combined = combined[~np.isnan(combined).any(1)]
    combined = combined.transpose()
    wanted_freq_range = combined[0]
    rolling_mean = combined[1]
    print('Done!')
    print("--- %s seconds ---" % (time.time() - start_time))

    
    
    """
    Try different fits to the resulting residual spectrum
    """


    """
    Initial guess at parameter values
    """
    
    
    gaussian_scale_guess = 10*max_power
    gaussian_mu_guess = nu_max
    gaussian_sd_guess = 270.0
    
    lorentzian_scale_guess = 10*max_power
    lorentzian_mu_guess = nu_max
    lorentzian_fwhm_guess = 400
    
    sgx0 = np.array([gaussian_scale_guess,gaussian_mu_guess,gaussian_sd_guess])
    slx0 = np.array([lorentzian_scale_guess,lorentzian_mu_guess,lorentzian_fwhm_guess])
    
    """
    Apply the least_squares fitting function
    """
    print('Gaussian parameter guess: ' , sgx0)
    print('Optimizing...')
    sgpopt,sgpcov = curve_fit(gaussian,wanted_freq_range,rolling_mean,p0=sgx0,bounds=(0,np.inf))
    print('Done!')
    print("--- %s seconds ---" % (time.time() - start_time))
    print('Gaussian parameter result: ' , sgpopt)
    
    
    print('Lorentzian parameter guess: ' , sgx0)
    print('Optimizing...')
    slpopt,slpcov = curve_fit(lorentzian,wanted_freq_range,rolling_mean,p0=slx0,bounds=(0,np.inf))
    print('Done!')
    print("--- %s seconds ---" % (time.time() - start_time))
    print('Lorentzian parameter result: ' , slpopt)
    
    
    return sgpopt
    
    




"""
This function fits the entire lomb-scargle periodogram using MCMC.
Due to computation-time considerations, the data is binned before it is fitted.
The error for the likelihood functions are then estimated using the standard
deviation of the binned data points and the additional smoothing standard 
deviation.
"""
def mcmc_fit(kic_number,activity,freq_input,power_input,oscillation_range,delta_nu,gsparams):
    freq = freq_input.copy()
    power = power_input.copy()
    
    """
    nu_nyq is used for eta_correction funciton, as specified in the background
    fitting function.
    """
    nu_nyq = np.amax(freq)
    
   
    """
    For the complex background model (if used).
    """
    def eta_correction(freq):
        nu_nyq = np.amax(freq)
        x = ((np.pi/2)*(freq/nu_nyq))
        return (np.sin(x)/x)**2
    
    
    """
    Returns a gaussian/lorentzian shape given an array and parameters.
    """
    def gaussian(x,scale,mu,sd):
        return scale*(1/np.sqrt(2*np.pi*sd**2))*np.exp(-0.5*((x-mu)/sd)**2)
    
    
    def lorentzian(x,scale,mu,width):
        return (scale/width)/(1 + (2*(x-mu)/width)**2)
    
    
    
        
    
    
    """
    Functions to be fitted to the data. Background model is given both for 
    complex and simple profiles as specified above when the background was fitted by 
    itself.
    signal_total is just a sum of the background and the oscillation shape (by 
    deafult, this is a gaussian shape)
    """
    
    
    """
    Complex Background functions
    """
    
    """
    if activity == True:
        def bg_total(freq,p_gran,tau_gran,a_gran,p_act,tau_act,b):       
            x=  eta_correction(freq)*(p_gran*tau_gran/((1+(tau_gran*freq)**a_gran)) + p_act*tau_act/(1+(tau_act*freq)**2)) + b
            return x
    else:
        def bg_total(freq,p_gran,tau_gran,a_gran,p_act,tau_act,b):       
            x=  eta_correction(freq)*(p_gran*tau_gran/((1+(tau_gran*freq)**a_gran))) + b
            return x
        
    
    
    def bg_gran(freq,p_gran,tau_gran,a_gran):
        x = eta_correction(freq)*(p_gran*tau_gran/ ((1+(tau_gran*freq)**a_gran)))
        return x
    
    def bg_act(freq,p_act,tau_act):
        x= eta_correction(freq)*(p_act*tau_act/(1+(tau_act*freq)**2))
        return x
    
    def bg_offset(freq,b):
        x = np.full(freq.size,b)
        return x
    """
    
    
    """
    Total power calculator
    """
    
    if activity == True:
        def signal_total(freq,p_gran,tau_gran,a_gran,p_act,tau_act,b,scale,mu,sd):
            return bg_total(freq,p_gran,tau_gran,a_gran,p_act,tau_act,b) + gaussian(freq,scale,mu,sd)
    else:
        def signal_total(freq,p_gran,tau_gran,a_gran,b,scale,mu,sd):
            return bg_total(freq,p_gran,tau_gran,a_gran,b) + gaussian(freq,scale,mu,sd)
    
    """
    Simple background functions
    """
    
    
    if activity == True:
        def bg_total(freq,p_gran,tau_gran,a,p_act,tau_act,b):
            x=  p_gran/((1+(tau_gran*freq)**a)) + p_act/(1+(tau_act*freq)**2) + b
            return x
        
    else:
         def bg_total(freq,p_gran,tau_gran,a,b):
            x=  p_gran/((1+(tau_gran*freq)**a)) + b
            return x
        
    
    def bg_gran(freq,p_gran,tau_gran,a):
        x = p_gran/ ((1+(tau_gran*freq)**a))
        return x
    
    def bg_act(freq,p_act,tau_act):
        x= p_act/(1+(tau_act*freq)**2)
        return x
    
    def bg_offset(freq,b):
        x = np.full(freq.size,b)
        return x
    
    
    
    """
    Functions used for MCMC. The log of the likelihood function, the prior distribution,
    and the overall function passed to the emcee EnsembleSampler, lnprob.
    theta is an array of the parameters of the fit.
    """
    def lnlike(theta,x,y,yerr,act_ind):
        if act_ind == True:
            p_gran,tau_gran,a_gran,p_act,tau_act,b,scale,mu,sd = theta
            model = signal_total(x,p_gran, tau_gran,a_gran,p_act,tau_act,b,scale,mu,sd)
        else:
            p_gran,tau_gran,a_gran,b,scale,mu,sd = theta
            model = signal_total(x,p_gran, tau_gran,a_gran,b,scale,mu,sd)
        sigma2 = yerr**2 
        squared_err = ((model-y)**2)/(2*sigma2)
        #A factor of 10 is used here to restrict the movements of the MCMC walkers further.
        return -10*np.sum(squared_err)
    
    
    def lnprior(theta,p0):
        #Top hat distribution for the fit parameters. We assume that the previously fitted
        #parameters are at least found to the correct order of magnitude, a reasonable
        #assumption.
        if all(0.1*p0[i] < theta[i] < 10*p0[i] for i in range(p0.size)):
            return 0.0
        return -np.inf
    
    def lnprob(theta, x, y,yerr,p0,act_ind):
        lp = lnprior(theta,p0)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(theta, x, y,yerr,act_ind)
    
    
    """
    Number of bins that data will be binned into. Also declare the power excess
    hump optimized parameters obtained by peak_fitting (see above).
    """
    nbins = 10000
    width = freq.size // nbins
    scale_guess = gsparams[0]
    mu_guess = gsparams[1]
    sd_guess = gsparams[2]
    fwhm_guess = gsparams[2]
    
    
    
    
    """
    Split the data into bins, and also obtain the standard deviation of the data
    used for each point.
    """
    print('Binning Data...')
    binned_freq = freq[:(freq.size // width) * width].reshape(-1, width).mean(axis=1)
    print('Binned frequency data points: ' , binned_freq.size)
    binned_power = power[:(power.size//width) * width].reshape(-1,width).mean(axis=1)
    print('Binned power data points: ' , binned_power.size)
    binned_power_err = power[:(power.size//width)*width].reshape(-1,width).std(axis=1)
    print('Binned power errors size: ' , binned_power_err.size)    
    print('Done!')
    print("--- %s seconds ---" % (time.time() - start_time))
    
    """
    Smooth the binned data using a boxcar filter of width 2*delta-nu.
    """
    
    
    idx_max = find_nearest(binned_freq,mu_guess)
    nu_max = binned_freq[idx_max]
    window_size = find_nearest(binned_freq,nu_max + 1.0*delta_nu) - find_nearest(binned_freq,nu_max - 1.0*delta_nu)
    
    
    print('Applying Heavy Smoothing...')
    
    
    series = Series(binned_power,index = binned_freq)
    rolling = series.rolling(window = window_size,center = True,min_periods = 100)
    rolling_mean = rolling.mean()
    rolling_mean = rolling_mean.to_numpy(dtype = float)
    rolling_std = rolling.std()
    rolling_std = rolling_std.to_numpy(dtype = float)
    
    binned_power_copy = rolling_mean
    binned_freq_copy = binned_freq.copy()
    
    combined = np.array([binned_freq_copy,binned_power_copy,binned_power_err,rolling_std])
    combined = combined.transpose()
    combined = combined[~np.isnan(combined).any(1)]
    combined = combined.transpose()
    binned_freq_copy = combined[0]
    binned_power_copy = combined[1]
    binned_power_err = combined[2]
    smoothing_power_err = combined[3]
    
    print('Done!')
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
    
    """
    Print out the error arrays for sanity check.
    """
    print('Binned power errors: ' ,  binned_power_err)
    print('Binned power errors size: ' ,  binned_power_err.size)
    print('Smoothing power errors: ' ,  smoothing_power_err)
    print('Smoothing power errors size: ' , smoothing_power_err.size)
    total_power_err = np.sqrt(binned_power_err**2 + smoothing_power_err**2)
    #total_power_err = binned_power_err
    
    
    """
    Guess the error in the readings of power to be used in the log-likelihood function.
    Currently set to 5% of the given reading, but can easily be adjusted to a constant value.
    This is an alternative to total_power_err.
    """
    #noise = 0.05*np.mean(binned_power)
    
    
    
    
    """
    Remove the oscillation signal data as it is not part of 
    the background. The data for low frequency values needs to be removed here
    if you need to ignore the activity part of the background.
    Also declare the initial fit parameter guesses.
    """
    
    if activity == True:
        power_exc_signal = np.delete(binned_power_copy,np.where((oscillation_range[0]<binned_freq_copy) & (binned_freq_copy <oscillation_range[1])))
        power_err_exc_signal = np.delete(total_power_err,np.where((oscillation_range[0]<binned_freq_copy) & (binned_freq_copy <oscillation_range[1])))
        freq_exc_signal = np.delete(binned_freq_copy,np.where((oscillation_range[0]<binned_freq_copy) & (binned_freq_copy <oscillation_range[1])))
        high_freq_lim = power_exc_signal[np.where(freq_exc_signal>1.0*oscillation_range[1])]
        """
        Complex background model parameter guesses
        """
        
        """
        p_gran_guess = np.max(power_exc_signal)*1e2
        tau_gran_guess = 1e-3
        tau_act_guess = 1e3
        p_act_guess = np.max(power_exc_signal)*1e2
        b_guess = np.mean(high_freq_lim)
        a_gran_guess = 2
        """
        
        
        """
        Simple background model parameter guesses
        """
        p_gran_guess = np.max(power_exc_signal)
        tau_gran_guess = 1e-3
        a_gran_guess = 2
        p_act_guess = np.max(power_exc_signal)
        tau_act_guess = 1e-1
        b_guess = np.mean(high_freq_lim)
        
        
        x0= np.array([p_gran_guess,tau_gran_guess,a_gran_guess,p_act_guess,tau_act_guess,b_guess,scale_guess,mu_guess,sd_guess])
        lsfit_x0 = np.array([p_gran_guess,tau_gran_guess,a_gran_guess,p_act_guess,tau_act_guess,b_guess])
    
    
    else:
        power_exc_signal = np.delete(binned_power_copy,np.where((oscillation_range[0]<binned_freq_copy) & (binned_freq_copy <oscillation_range[1])))
        power_exc_signal = np.delete(power_exc_signal,np.where(binned_freq_copy<100))
        power_err_exc_signal = np.delete(total_power_err,np.where((oscillation_range[0]<binned_freq_copy) & (binned_freq_copy <oscillation_range[1])))
        power_err_exc_signal = np.delete(power_err_exc_signal,np.where(binned_freq_copy<100))
        freq_exc_signal = np.delete(binned_freq_copy,np.where((oscillation_range[0]<binned_freq_copy) & (binned_freq_copy <oscillation_range[1])))
        freq_exc_signal = np.delete(freq_exc_signal,np.where(freq_exc_signal<100))
        high_freq_lim = power_exc_signal[np.where(freq_exc_signal>1.0*oscillation_range[1])]
        
        binned_power_copy = np.delete(binned_power_copy,np.where(binned_freq_copy<100))
        binned_power_err = np.delete(binned_power_err,np.where(binned_freq_copy<100))
        total_power_err = np.delete(total_power_err,np.where(binned_freq_copy<100))
        binned_freq_copy = np.delete(binned_freq_copy,np.where(binned_freq_copy<100))
        
        """
        Complex background model parameter guesses
        """
        
        """
        p_gran_guess = np.max(power_exc_signal)*1e2
        tau_gran_guess = 1e-3
        b_guess = np.mean(high_freq_lim)
        a_gran_guess = 2
        """
        
        
        """
        Simple background model parameter guesses
        """
        p_gran_guess = np.max(power_exc_signal)
        tau_gran_guess = 1e-3
        a_gran_guess = 2
        b_guess = np.mean(high_freq_lim)
    
        x0= np.array([p_gran_guess,tau_gran_guess,a_gran_guess,b_guess,scale_guess,mu_guess,sd_guess])
        lsfit_x0 = np.array([p_gran_guess,tau_gran_guess,a_gran_guess,b_guess])
   
    
    
    
    
   
    print('Initial Parameter Guess: ' , x0)
    
    """
    Optimize the backrgound parameters using curve_fit to get a better initial guess 
    at their values to be passed to MCMC.
    """
    print('Optimizing Initial Parameter Guess...')
    popt,pcov = curve_fit(bg_total,freq_exc_signal,power_exc_signal,p0=lsfit_x0,\
                          sigma = power_err_exc_signal, bounds=(0,np.inf))
    
    if activity == True:
        p0 = np.array([popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],scale_guess,mu_guess,sd_guess])
    else:
        p0 = np.array([popt[0],popt[1],popt[2],popt[3],scale_guess,mu_guess,sd_guess])

    print('Done!')
    print('Curve_Fit parameters: ' , p0)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
    """
    Set up the parameters of the MCMC algorithm and the initial position of the walkers, pos.
    ndim is the number of fit parameters.
    nwalkers is the number of walkers.
    niter is the number of iterations the MCMC sampler runs through.
    nburn is the burn-in, i.e the number of initial steps that will be discarded.
    niter must be larger than nburn.
    """
    ndim = x0.size
    nwalkers = 100
    nburn = 1000
    niter = 2000
    pos = [p0 + 1e-3*p0*np.random.uniform(-1.0,1.0) for i in range(nwalkers)]
     


    """
    Run the MCMC algorithm.
    """
    
    print('Running MCMC algorithm...')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(binned_freq_copy, binned_power_copy, total_power_err,p0,activity))
    sampler.run_mcmc(pos, niter)
    print('Done!')
    print("--- %s seconds ---" % (time.time() - start_time))
    samples = sampler.chain
    print('Acceptance Fraction of Moves')
    print(sampler.acceptance_fraction)
    
    
    
    """
    Obtain the 16,50, and 84 percentile values for each parameter after
    the burn-in period.
    """
    
    if activity == True:
        samples2 = samples[:,nburn:,:].reshape((-1,ndim))
        p_gran,tau_gran,a_gran,p_act,tau_act,b,scale,mu,sd = [np.percentile(samples2[:,i],[16,50,84]) for i in range(ndim)]
        print('Percentile Parameter Values...')
        print('p_gran: ' , p_gran)
        print('tau_gran: ' , tau_gran)
        print('a_gran: ' , a_gran)
        print('p_act: ' , p_act)
        print('tau_act: ' , tau_act)
        print('b: ' , b)
        print('gaussian scale: ' , scale)
        print('gaussian mu: ' , mu)
        print('gaussian sd: ' , sd)
    else:
        samples2 = samples[:,nburn:,:].reshape((-1,ndim))
        p_gran,tau_gran,a_gran,b,scale,mu,sd = [np.percentile(samples2[:,i],[16,50,84]) for i in range(ndim)]
        print('Percentile Parameter Values...')
        print('p_gran: ' , p_gran)
        print('tau_gran: ' , tau_gran)
        print('a_gran: ' , a_gran)
        print('b: ' , b)
        print('gaussian scale: ' , scale)
        print('gaussian mu: ' , mu)
        print('gaussian sd: ' , sd)
    
    
    
    
    
    """
    Obtain the best MCMC fit parameters
    """
    samples3 = sampler.flatchain
    res = samples3[np.argmax(sampler.flatlnprobability)]
    print('Best MCMC fit parameters:')
    print(res)
    

    
    """
    Return the best fit parameters, along with the binned data.
    """
    return res,binned_freq,binned_power,total_power_err
    

"""
This function calculates the maximum mode amplitude using the method described by
Kjeldsen et al. (2008b) in https://arxiv.org/pdf/0804.1182.pdf
The data is smoothed using a gaussian window of FWHM of 4*delta-nu. Then it is scaled
by a factor of delta-nu / 3.04. 
The average around the central value of width 3*delta_nu is then used as the quoted 
figure.
"""
def max_amplitude_finder(kic_number,activity,freq_input,power_input,fit_params,delta_nu,oscillation_range):
    freq = freq_input.copy()
    power = power_input.copy()
    res = fit_params
    
    """
    Return Gaussian/Lorentzian shape.
    """
    
    def gaussian(x,scale,mu,sd):
        return scale*(1/np.sqrt(2*np.pi*sd**2))*np.exp(-0.5*((x-mu)/sd)**2)
    
    def lorentzian(x,scale,mu,width):
        return (scale/width)/(1 + (2*(x-mu)/width)**2)
    
    """
    Total power calculator
    """
    
    if activity == True:
        def signal_total(freq,p_gran,tau_gran,a_gran,p_act,tau_act,b,scale,mu,sd):
            return bg_total(freq,p_gran,tau_gran,a_gran,p_act,tau_act,b) + gaussian(freq,scale,mu,sd)
        def bg_total(freq,p_gran,tau_gran,a,p_act,tau_act,b):
            x=  p_gran/((1+(tau_gran*freq)**a)) + p_act/(1+(tau_act*freq)**2) + b
            return x
        """
        def bg_total(freq,p_gran,tau_gran,a_gran,p_act,tau_act,b):
            x=  eta_correction(freq)*(p_gran*tau_gran/((1+(tau_gran*freq)**a_gran)) + p_act*tau_act/(1+(tau_act*freq)**2)) + b
        return x
        """
    else:
        def signal_total(freq,p_gran,tau_gran,a_gran,b,scale,mu,sd):
            return bg_total(freq,p_gran,tau_gran,a_gran,b) + gaussian(freq,scale,mu,sd)
        def bg_total(freq,p_gran,tau_gran,a,b):
            x=  p_gran/((1+(tau_gran*freq)**a)) + b
            return x
        """
        def bg_total(freq,p_gran,tau_gran,a_gran,b):
            x=  eta_correction(freq)*(p_gran*tau_gran/((1+(tau_gran*freq)**a_gran))) + b
        return x
        """
        

    
    """
    Smoothing Function
    """
    
    def smooth(power_array,window_size,sd):
        print('Window size: ' , window_size)
        print('Standard Deviation : ', sd)
        series = Series(power_array,index = freq)
        rolling = series.rolling(window = window_size,win_type = 'gaussian' , center = True,min_periods = 100)
        rolling_mean = rolling.mean(std = sd)
        rolling_mean = rolling_mean.to_numpy(dtype = float)
        return rolling_mean
    
        
    
    """
    Smooth the data and remove NaN values.
    """
    fwhm_size = find_nearest(freq,3000+2*delta_nu) - find_nearest(freq,3000-2*delta_nu)   
    sd = fwhm_size / (2*np.sqrt(2*np.log(2)))
    window_size = 2*fwhm_size
    smoothed_power = smooth(power,window_size,sd)
    combined = np.array([freq,smoothed_power,power])
    combined = combined.transpose()
    combined = combined[~np.isnan(combined).any(1)]
    combined = combined.transpose()
    
    freq = combined[0]
    smoothed_power = combined[1]
    power = combined[2]
    
    
    """
    Compute the central window and the maximum mode amplitude.
    """
    
    mode_amp_freq_range = freq[(freq>res[res.size-2]-1.5*delta_nu)&(freq<res[res.size-2] + 1.5*delta_nu)]
    mode_amp_power_range = smoothed_power[(freq>res[res.size-2]-1.5*delta_nu)&(freq<res[res.size-2] + 1.5*delta_nu)]
    if activity == True:
        res_power = mode_amp_power_range - bg_total(mode_amp_freq_range,res[0],res[1],res[2],res[3],res[4],res[5])
    else:
        res_power = mode_amp_power_range - bg_total(mode_amp_freq_range,res[0],res[1],res[2],res[3])
    res_power_density = res_power*delta_nu/3.04
    amp_density = np.sqrt(res_power_density)
    
    print('Maximum Amplitude: ')
    envmax = np.amax(amp_density)
    print(np.amax(amp_density))
    print('Average Amplitude: ')
    envmean = np.mean(amp_density)
    print(np.mean(amp_density))
    
    """
    Return the mode amplitudes.
    """
    return envmax,envmean  




"""
The main function passes through all the previously defined functions in order.
The only argument needed is the filename where the timeseries data is contained,
and a boolean value to declare whether activity is to be considered in the background
or not.
"""
def main(fits_filename,activity):
    
    """
    Obtain the KIC number of the star. Note that the filename is saved in the original
    KASC naming convention, so that the 9 digit number is contained between the 4th and 12th
    characters of the string inclusive.
    """
    kic_number = fits_filename[4:13]
    print(kic_number)
    
    print('Running Oscillation Range Finder...')
    freq,power,oscillation_range,sph = finding_freq_range(kic_number,fits_filename)
    print('Done!')
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Smoothing...')
    smoothed_freq,smoothed_power,smoothed_power_err = smoothing(kic_number,freq,power)
    print('Done!')
    print("--- %s seconds ---" % (time.time() - start_time))


    print('Creating Background Model...')
    if activity == True:
        reduced_freq,reduced_power,bg_gran_power,bg_act_power,\
        bg_offset_power,bg_total_power = total_background(kic_number,activity,smoothed_freq,\
                                                          smoothed_power,smoothed_power_err,\
                                                          oscillation_range)
    else:
        reduced_freq,reduced_power,bg_gran_power,\
        bg_offset_power,bg_total_power = total_background(kic_number,activity,smoothed_freq,\
                                                          smoothed_power,smoothed_power_err,\
                                                          oscillation_range)
    print('Done!')
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Finding Peak Spacing...')
    delta_nu = find_delta_nu(kic_number,reduced_freq,reduced_power,bg_total_power,oscillation_range)
    print('Done!')
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Fitting Oscillation Parameters...')
    gsparams = peak_fitting(kic_number,reduced_freq,reduced_power,bg_total_power,oscillation_range,delta_nu)
    print('Done!')
    print("--- %s seconds ---" % (time.time() - start_time))
    
    print('Preparing MCMC...')
    res,binned_freq,binned_power,total_power_err = mcmc_fit(kic_number,activity,freq,power,oscillation_range,delta_nu,gsparams)
    print('Done!')
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
    print('Calculating Mode Amplitude...')
    envmax,envmean =max_amplitude_finder(kic_number,activity,binned_freq,binned_power,res,delta_nu,oscillation_range)
    
    print('Summary of Parameters: ')
    print('Mu: ' , res[res.size-2])
    print('Delta nu: ' , delta_nu)
    print('Mean Amplitude Envelope: ' , envmean)
    print('SPH: ', sph)

    #res[res.size-2] will, regardless of whether activity is included or not, contain
    #the optimized value for the mean of the gaussian excess power hump.
    return kic_number,oscillation_range,res[res.size-2],delta_nu,envmean,sph
    #plt.show()




"""
Declare an empty table, and loop through the timeseries files for the stars
given. If the code produces results for a star, save the results into a new row in
the table. If not, skip to the next star (you can change this to do something
else by modifying the code after the "except" clause).
"""
table = []
fields = ['KIC No.' , 'Oscillation Range', 'Mu','Delta Nu','Env,Mean','SPH']
table.append(fields)
directory = os.fsencode(fits_directory)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    try:
        
        #Change True/False based on whether keeping activity (True) or discarding it
        #(false).
        kic_number,oscillation_range,mu,delta_nu,envmean,sph = main(filename,False)
        newrow = [kic_number,oscillation_range,mu,delta_nu,envmean,sph]
        table.append(newrow)
        print(newrow)
    except:
        print('Error encountered, skipping to next star...')
        pass

"""
Finally, print the results, using "tabulate" to make the formatting easier to 
read, and save to a  .txt file using pickle.
"""
print('Final Results')
print(table)

table2 = tabulate((table[i] for i in range(1,len(table))),headers=table[0])
print(table2)
with open(text_directory+'amplitudetable_v6.txt','wb') as f:
    pickle.dump(table,f)
    
