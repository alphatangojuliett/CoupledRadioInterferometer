import pyuvdata.utils as uvutils
import sys
import numpy as np
import os
from pyuvdata import UVData, UVBeam
import healpy as hp
from astropy.constants import c
import numpy.polynomial.chebyshev as cheby

#include locally-revised Github code:
sys.path.insert(1, '/home/atj/Github_Repos/local_edits/') # insert at 1, 0 is the script path (or '' in REPL)

c_ms = c.to("m/s").value

import healvis_with_coupling as hwc

def find_keys(bl, antpos_metadata, ants_metadata, bi=1):
    ant_i_enu=bl.ant1_enu
    ant_j_enu=bl.ant2_enu
    ii_anti_antpos = np.nonzero([np.all(antpos == ant_i_enu) for antpos in antpos_metadata])[0][0]
    ii_antj_antpos = np.nonzero([np.all(antpos == ant_j_enu) for antpos in antpos_metadata])[0][0]
    anti=ants_metadata[ii_anti_antpos]
    antj=ants_metadata[ii_antj_antpos]
    key_Vij1 = tuple((anti, antj))
    baseline_number_ij = pyuvutils.antnums_to_baseline(anti,antj, Nants_telescope=Nants_used)
    baseline_number_ji = pyuvutils.antnums_to_baseline(antj,anti, Nants_telescope=Nants_used)
    ii_bl_group = np.nonzero([np.logical_or(baseline_number_ij in group, baseline_number_ji in group)for group in bl_groups])[0][0]
    key_Vij0=tuple(pyuvutils.baseline_to_antnums(bl_groups[ii_bl_group][0], Nants_telescope=Nants_used))

    return key_Vij1, key_Vij0


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def find_any_Vij0_with_same_ENU(ant_1_enu, ant_2_enu, antpos_metadata, ants_metadata, dict_Vij0s,vec_bin_centers,tol=0.6):
    ENU_12 = np.array(ant_2_enu-ant_1_enu)
    ENU_21 = np.array(ant_1_enu-ant_2_enu)
    
    ii_ant1_antpos = np.nonzero([np.all(antpos == ant_1_enu) for antpos in antpos_metadata])[0][0]
    ii_ant2_antpos = np.nonzero([np.all(antpos == ant_2_enu) for antpos in antpos_metadata])[0][0]
    ant1=ants_metadata[ii_ant1_antpos]
    ant2=ants_metadata[ii_ant2_antpos]

    baseline_number_12 = pyuvutils.antnums_to_baseline(ant1,ant2, Nants_telescope=Nants_used)
    baseline_number_21 = pyuvutils.antnums_to_baseline(ant2,ant1, Nants_telescope=Nants_used)

    
    ii_bl_group = np.nonzero([ np.logical_or(baseline_number_12 in group, baseline_number_21 in group) for group in bl_groups])[0][0]
    key_Vij0=tuple(pyuvutils.baseline_to_antnums(bl_groups[ii_bl_group][0], Nants_telescope=Nants_used))
    
    if np.all(np.abs(ENU_12-vec_bin_centers[ii_bl_group])<=tol):
        return dict_Vij0s[key_Vij0]
    elif np.all(np.abs(ENU_21-vec_bin_centers[ii_bl_group])<=tol):
        return np.conjugate(dict_Vij0s[key_Vij0])
    else:
        print('WARNING in find_any_Vij0_with_same_ENU(): Potential conjugation Error? OR out of tolerance.')
    
    
def exp(theta):
    exp=np.cos(theta) + 1j * np.sin(theta)
    return exp


def make_new_coupling_beams(dict_coupling):

    uvbE = UVBeam()
    uvbD = UVBeam()
    uvbE.read_beamfits(dict_coupling['str_E'])
    uvbD.read_beamfits(dict_coupling['str_D'])

    print('Calculating HH...')
    uvbHH=hwc.beam_model.E_to_HH_orthonormal(uvbE, uvbD, inplace=False)
    print('Done. HH polarization array: '+str(uvbHH.polarization_array))
    print('HH x_orientation: '+str(uvbHH.x_orientation))
    print('Writing HH beamfits...')
    uvbHH.write_beamfits(dict_coupling['str_HH'], run_check=False, clobber=True)
    print('    done.')

    print('Calculating HdagHdag...')  
    uvbHdagHdag=hwc.beam_model.E_to_HdagHdag_orthonormal(uvbE, uvbD, inplace=False)
    print('Done. Writing HdagHdag beamfits...')
    uvbHdagHdag.write_beamfits(dict_coupling['str_HdagHdag'], run_check=False, clobber=True)
    print('    done.')

    return uvbHH, uvbHdagHdag


def calc_first_order_visibility_uvdata(uvd0, uvd1_dict, dict_coupling):
    """
    az, za = Azimuth, zenith angle, radians
    freq = frequeny in Hz
    enu = 'ij'th baseline vector in meters
    uvd0 : pyuvdata object, zeroth order visibilities
    uvdHD1: the heraData object which will contain the 1st order visibilities
    dict_coupling: dictionary generated from the 1st order coupling YAML file. 
    """

    OUT={}

    use_keys_out = dict_coupling['use_keys_out']

    #generate feed_freq_gamma_resistance
    gamma_path=dict_coupling['gamma_path']
    feed_gamma_coeff=np.load(os.path.join(gamma_path, dict_coupling['gamma_fname']))
    feed_resistance=np.loadtxt(os.path.join(gamma_path, dict_coupling['z_ant_real']))
    freqs_gamma_and_resistance=np.loadtxt(os.path.join(gamma_path, dict_coupling['f_real']))
    feed_freq_gamma_resistance = np.vstack((freqs_gamma_and_resistance*1e6, feed_gamma_coeff, feed_resistance)) #frequency should emter array in units [Hz], not MHz

    #define impedance and resistance parameters 
    freq=uvd0.freq_array[0]
    freqs_gamma_and_resistance = feed_freq_gamma_resistance[0,:] #the freqencies associated with gamma, R_ant, in Hz
    ii_fgamma_for_freq = np.array([find_nearest_index(freqs_gamma_and_resistance.real, f) for f in freq]).astype(int)
    gamma =  feed_freq_gamma_resistance[1,ii_fgamma_for_freq]# to come from CST models of the HERA beam, is the voltage reflection coefficient. It is the attenuation factor of re-radiated emission from one antenna to another based on impedance mismatch (a la http://faculty.nps.edu/jenn/pubs/flokasAPS.pdf), for actual (frequency dependent): https://arxiv.org/pdf/2009.07939.pdf
    R_ant = feed_freq_gamma_resistance[2,ii_fgamma_for_freq] #[Ohms], the real part of radiation impedance, which again is discussed in flokas and given as a function of frequency in Fagnoni et al.
    
    # Perform a Chebyshev polynomial fit on gamma and R_ant, to avoid artifacts in the power spectrum caused by array indexing
    gammaR = gamma.real 
    gammaI = gamma.imag

    gammaR_Cheby = cheby.Chebyshev.fit(freq, gammaR, chebydeg)
    gammaI_Cheby = cheby.Chebyshev.fit(freq, gammaI, chebydeg)

    gamma_Cheby = gammaR_Cheby(freq)+(1j)*gammaI_Cheby(freq)

    R_Cheby = cheby.Chebyshev.fit(freq, R_ant, chebydeg)
    
    #Load HH and HdagHdag beams
    if dict_coupling['make_new_coupling_beams']: #Option to make new coupling beams (HH, HdagHdag)
        uvbHH, uvbHdagHdag = make_new_coupling_beams(dict_coupling)
    else:
        print('    Reading in coupled beams...')
        HH_beam = UVBeam()
        HH_beam.read_beamfits(dict_coupling['str_HH'])

        HdagHdag_beam = UVBeam()
        HdagHdag_beam.read_beamfits(dict_coupling['str_HdagHdag'])
        print('    Done.')

    za, az = hp.pixelfunc.pix2ang(HH_beam.nside, np.arange(HH_beam.Npixels))

    pol = dict_coupling['pol']
    same_beam_all_freqs = dict_coupling['same_beam_all_freqs']
    antpos, ants = uvd0.get_ENU_antpos(pick_data_ants=False)
    #ANTPAIRPOL LOOP STARTED JUST BELOW
    
    print('    Looping over antpairpols to couple...')
    antpairpols_to_couple = uvd0.get_antpairpols()
    if use_keys_out:
        print('    Using keys out')
        antpairpols_to_couple = dict_coupling['keys_out']
    else: 
        print('    Applying coupling to all antpairpols in the uvdata object')

    iantpair=0
    for antpairpol in antpairpols_to_couple: #dict_coupling['keys_out']:     

        #ANTPAIRPOL LOOP DID START JUST ABOVE

        antpairpol=tuple(antpairpol)

        ant_i_enu = antpos[np.nonzero(antpairpol[0]==ants)[0][0]]
        ant_j_enu = antpos[np.nonzero(antpairpol[1]==ants)[0][0]]
        Ai = ants[np.argwhere([np.all(np.isclose(ant_i_enu, ENU)) for ENU in antpos])[0][0]]
        Aj = ants[np.argwhere([np.all(np.isclose(ant_j_enu, ENU)) for ENU in antpos])[0][0]]
        enu= ant_j_enu - ant_i_enu
        baseline_number = uvd0.antnums_to_baseline(antpairpol[0],antpairpol[1])
        Vij0 = uvd0.get_data(antpairpol)#Vij0s[antpairpol]
        if use_keys_out: print('    antpairpol (Vij0): '+str(antpairpol))

        

        coupling_sigma_knotj = 0+0j #used only if coupling_order = 1
        coupling_sigma_knoti = 0+0j

        coupling_constants = (freq/c_ms)*(1./4.)*1j 
        #WARNING, TEMPORARY, SINCE NOT USING HEALVIS.BEAM_MODEL.BEAM_VAL for HH and HdagHdag
        ipol=[]
        if pol == 'XX' or pol =='xx': 
            ipol=0
        elif pol== 'YY' or pol =='yy': 
                ipol=1
        else:
            print('    ERROR when calculating first order visibility. invalid pol. only use diagonals for now, when using HH HdagHdag.')
            return
        
        za_bl = np.pi/2 #The zenith angle of all baselines is 90 degrees = pi/2
        ii_V_freqs_in_beam_array=[]

        iants_knoti=0
        iants_knotj=0
        ants_coupled = [] #keep track of all unique Ak that are used in coupling.
        
        if dict_coupling['limit_coupling']['type']!='none': #only loop if there are ants to couple.
            for k in antpos:
                Ak = ants[np.argwhere([np.all(np.isclose(k, ENU)) for ENU in antpos])[0][0]] #to get the antenna number from antenna ENU, 
                if np.linalg.norm(k-ant_i_enu ) !=0 : #k is not the ith antenna
                    #Calculate azimuth KI-hat from ENU
                    enu_ki = np.array(ant_i_enu-k)
                    az_bl = np.arctan(enu_ki[1]/enu_ki[0])#radians
                    
                    pix_az_bl_at_horizon = hp.ang2pix(HH_beam.nside,za_bl, az_bl) 
                    HH_at_bl = HH_beam.data_array[0,0,ipol,:,pix_az_bl_at_horizon] #one pixel, for all BEAM frequencies , if beam val: HH_cube[ii_az_bl_at_horizon, :]
                    
                    if same_beam_all_freqs:
                        ii_V_freqs_in_beam_array = np.array([105 for f_simulated in freq]).astype(int) 
                    else:
                        ii_V_freqs_in_beam_array = np.array([find_nearest_index(HH_beam.freq_array[0], f_simulated) for f_simulated in freq]).astype(int) #assumes Hz, both freq arrays
                    HH_at_bl = HH_at_bl[ii_V_freqs_in_beam_array] 

                    #calculate zeroth order visibility, kj
                    enu_kj = np.array(ant_j_enu-k)                
                    Vkj0= uvd0.get_data((Ak,Aj, dict_coupling['pol'])) # any redundant baseline with same ENU will work. BUT should it be (Ak,Aj) or (Aj, Ak) sticking with convention enu_kj= k->j // verified by uvw_array convention https://pyuvdata.readthedocs.io/en/latest/uvdata_parameters.html)


                    #calculate ki terms
                    propagator_dir= 1.
                    if dict_coupling['alternate_coupling']['flip_propagator_dir']: propagator_dir= -1.
                    coupling_ki = ((coupling_constants*gamma_Cheby)/np.linalg.norm(enu_ki))*exp( (propagator_dir*2. * np.pi * (np.linalg.norm(enu_ki)/(c_ms / freq))) )

                    #Limit certain baselines, based on visibility term not delay term
                    E = enu_kj[0]; N = enu_kj[1]; ABS = np.linalg.norm(enu_kj);
                    if dict_coupling['limit_coupling']['type'] == 'thresh':
                        if eval(dict_coupling['limit_coupling']['E/W']) and eval(dict_coupling['limit_coupling']['N/S']) and eval(dict_coupling['limit_coupling']['ABS']): # execute thresholds from YAML
                            iants_knoti+=1
                            ants_coupled.append(Ak)
                            
                            coupling_sigma_knoti = coupling_sigma_knoti + coupling_ki*HH_at_bl*Vkj0
                    elif dict_coupling['limit_coupling']['type'] == 'manual' and dict_coupling['IJK_manual_coupling'] == tuple((Ai,Aj,Ak)):
                        print('    In sigma_knoti, using manual IJK coupling for: '+str( tuple((Ai,Aj,Ak)) ))
                        iants_knoti+=1
                        ants_coupled.append(Ak)
                        coupling_sigma_knoti = coupling_sigma_knoti + coupling_ki*HH_at_bl*Vkj0
                    elif dict_coupling['limit_coupling']['type'] == 'all':
                        iants_knoti+=1
                        ants_coupled.append(Ak)
                        coupling_sigma_knoti = coupling_sigma_knoti + coupling_ki*HH_at_bl*Vkj0

                if np.linalg.norm(k-ant_j_enu ) !=0 : #k is not the jth antenna
                    #Calculate azimuth KJ-hat from ENU
                    enu_kj = np.array(ant_j_enu-k)
                    az_bl = np.arctan(enu_kj[1]/enu_kj[0]) #radians
                    
                    pix_az_bl_at_horizon = hp.ang2pix(HdagHdag_beam.nside,za_bl, az_bl) 
                    HdagHdag_at_bl = HdagHdag_beam.data_array[0,0,ipol,:,pix_az_bl_at_horizon] #one pixel, for all BEAM frequencies , if beam val: #HdagHdag_cube[ii_az_bl_at_horizon, :] 
                    
                    if same_beam_all_freqs:
                        ii_V_freqs_in_beam_array = np.array([105 for f_simulated in freq]).astype(int) 
                    else:
                        ii_V_freqs_in_beam_array = np.array([find_nearest_index(HdagHdag_beam.freq_array[0], f_simulated) for f_simulated in freq]).astype(int) #assumes Hz, both freq arrays
                    HdagHdag_at_bl = HdagHdag_at_bl[ii_V_freqs_in_beam_array]  

                    #calculate zeroth order visibility, ik
                    enu_ik = np.array(k-ant_i_enu)
                    Vik0= uvd0.get_data((Ai,Ak,dict_coupling['pol'])) #Vij0s[(Ai,Ak,dict_coupling['pol'])]

                    #calculate kj terms
                    propagator_dir= -1.
                    if dict_coupling['alternate_coupling']['flip_propagator_dir']: propagator_dir= 1.
                    coupling_kj = ((coupling_constants*gamma_Cheby)/np.linalg.norm(enu_kj))*exp( (propagator_dir*2. * np.pi * (np.linalg.norm(enu_kj)/(c_ms / freq))) ) #yes, +2pi, not -2pi, in the exponent
                    
                    #Limit certain baselines, based on visibility term not delay term
                    E = enu_ik[0]; N = enu_ik[1]; ABS = np.linalg.norm(enu_ik);
                    if dict_coupling['limit_coupling']['type'] == 'thresh':

                        if eval(dict_coupling['limit_coupling']['E/W']) and eval(dict_coupling['limit_coupling']['N/S']) and eval(dict_coupling['limit_coupling']['ABS']): # execute thresholds from YAML
                            ants_coupled.append(Ak)
                            iants_knotj+=1
                            coupling_sigma_knotj = coupling_sigma_knotj + coupling_kj*Vik0*HdagHdag_at_bl
                    elif dict_coupling['limit_coupling']['type'] == 'manual' and dict_coupling['IJK_manual_coupling'] == tuple((Ai,Aj,Ak)):
                        iants_knotj+=1
                        ants_coupled.append(Ak)
                        coupling_sigma_knotj = coupling_sigma_knotj + coupling_kj*Vik0*HdagHdag_at_bl
                    elif dict_coupling['limit_coupling']['type'] == 'all':
                        iants_knotj+=1
                        ants_coupled.append(Ak)
                        coupling_sigma_knotj = coupling_sigma_knotj + coupling_kj*Vik0*HdagHdag_at_bl
            
        if dict_coupling['alternate_coupling']['sum_both_sigma']:
            Vij1 = Vij0 + coupling_sigma_knoti + coupling_sigma_knotj
        if dict_coupling['alternate_coupling']['subtr_both_sigma']:
            Vij1 = Vij0 - coupling_sigma_knoti - coupling_sigma_knotj
        elif dict_coupling['alternate_coupling']['flip_sigma_sum']:
            Vij1 = Vij0 + coupling_sigma_knoti - coupling_sigma_knotj
        else:
            Vij1 = Vij0 - coupling_sigma_knoti + coupling_sigma_knotj

        uvd1_dict[antpairpol]=Vij1

        if use_keys_out:
            print('  coupled ants for antpairpol('+str(antpairpol)+'): '+str(ants_coupled))
            print('    For antpairpol '+str(antpairpol)+', Ants looped over for k!=i : '+str(iants_knoti)+', k!=j : '+str(iants_knotj))
        else: 
            
            if iantpair%100==0:
                print('  iantpair: '+str(iantpair)+' / '+str(np.shape(antpairpols_to_couple)[0]))
            iantpair+=1


    #THIS MARKED END OF ANTPAIRPOL LOOP

    print('    Done.')
    return uvd1_dict


def calc_first_order_visibility_uvdata_mt(uvd0, dict_coupling, antpairpol, chebydeg=6):
    #A MULTITHREADED VERSION OF THE FUNCTION calc_first_order_visibility_uvdata
    """
    az, za = Azimuth, zenith angle, radians
    freq = frequeny in Hz
    enu = 'ij'th baseline vector in meters
    uvd0 : pyuvdata object, zeroth order visibilities
    uvdHD1: the heraData object which will contain the 1st order visibilities
    dict_coupling: dictionary generated from the 1st order coupling YAML file. 
    """

    use_keys_out = dict_coupling['use_keys_out']

    #generate feed_freq_gamma_resistance
    gamma_path=dict_coupling['gamma_path']
    feed_gamma_coeff=np.load(os.path.join(gamma_path, dict_coupling['gamma_fname']))
    feed_resistance=np.loadtxt(os.path.join(gamma_path, dict_coupling['z_ant_real']))
    freqs_gamma_and_resistance=np.loadtxt(os.path.join(gamma_path, dict_coupling['f_real']))
    feed_freq_gamma_resistance = np.vstack((freqs_gamma_and_resistance*1e6, feed_gamma_coeff, feed_resistance)) #frequency should emter array in units [Hz], not MHz

    #define impedance and resistance parameters 
    freq=uvd0.freq_array[0]
    freqs_gamma_and_resistance = feed_freq_gamma_resistance[0,:] #the freqencies associated with gamma, R_ant, in Hz
    ii_fgamma_for_freq = np.array([find_nearest_index(freqs_gamma_and_resistance.real, f) for f in freq]).astype(int)
    gamma =  feed_freq_gamma_resistance[1,ii_fgamma_for_freq]# to come from CST models of the HERA beam, is the voltage reflection coefficient. It is the attenuation factor of re-radiated emission from one antenna to another based on impedance mismatch (a la http://faculty.nps.edu/jenn/pubs/flokasAPS.pdf), for actual (frequency dependent): https://arxiv.org/pdf/2009.07939.pdf
    R_ant = feed_freq_gamma_resistance[2,ii_fgamma_for_freq] #[Ohms], the real part of radiation impedance, which again is discussed in flokas and given as a function of frequency in Fagnoni et al.

    # Perform a Chebyshev polynomial fit on gamma and R_ant, to avoid artifacts in the power spectrum caused by array indexing
    gammaR = gamma.real 
    gammaI = gamma.imag

    gammaR_Cheby = cheby.Chebyshev.fit(freq, gammaR, chebydeg)
    gammaI_Cheby = cheby.Chebyshev.fit(freq, gammaI, chebydeg)

    gamma_Cheby = gammaR_Cheby(freq)+(1j)*gammaI_Cheby(freq)

    R_Cheby = cheby.Chebyshev.fit(freq, R_ant, chebydeg)
    
    #Load HH and HdagHdag beams must now be outside the multithreaded loop
    #if dict_coupling['make_new_coupling_beams']: #Option to make new coupling beams (HH, HdagHdag)
    #    uvbHH, uvbHdagHdag = make_new_coupling_beams(dict_coupling)
    #else:
    HH_beam = UVBeam()
    HH_beam.read_beamfits(dict_coupling['str_HH'])

    HdagHdag_beam = UVBeam()
    HdagHdag_beam.read_beamfits(dict_coupling['str_HdagHdag'])

    za, az = hp.pixelfunc.pix2ang(HH_beam.nside, np.arange(HH_beam.Npixels))

    pol = dict_coupling['pol']
    same_beam_all_freqs = dict_coupling['same_beam_all_freqs']
    antpos, ants = uvd0.get_ENU_antpos(pick_data_ants=False)


    antpairpol=tuple(antpairpol)

    ant_i_enu = antpos[np.nonzero(antpairpol[0]==ants)[0][0]]
    ant_j_enu = antpos[np.nonzero(antpairpol[1]==ants)[0][0]]
    Ai = ants[np.argwhere([np.all(np.isclose(ant_i_enu, ENU)) for ENU in antpos])[0][0]]
    Aj = ants[np.argwhere([np.all(np.isclose(ant_j_enu, ENU)) for ENU in antpos])[0][0]]
    enu= ant_j_enu - ant_i_enu
    baseline_number = uvd0.antnums_to_baseline(antpairpol[0],antpairpol[1])
    Vij0 = uvd0.get_data(antpairpol)#Vij0s[antpairpol]
    if use_keys_out: print('    antpairpol (Vij0): '+str(antpairpol))

    

    coupling_sigma_knotj = 0+0j #used only if coupling_order = 1
    coupling_sigma_knoti = 0+0j

    coupling_constants = (freq/c_ms)*(1./4.)*1j 
    #WARNING, TEMPORARY, SINCE NOT USING HEALVIS.BEAM_MODEL.BEAM_VAL for HH and HdagHdag
    ipol=[]
    if pol == 'XX' or pol =='xx': 
        ipol=0
    elif pol== 'YY' or pol =='yy': 
            ipol=1
    else:
        print('    ERROR when calculating first order visibility. invalid pol. only use diagonals for now, when using HH HdagHdag.')
        return
    
    za_bl = np.pi/2 #The zenith angle of all baselines is 90 degrees = pi/2
    ii_V_freqs_in_beam_array=[]

    iants_knoti=0
    iants_knotj=0
    ants_coupled = [] #keep track of all unique Ak that are used in coupling.
    
    if dict_coupling['limit_coupling']['type']!='none': #only loop if there are ants to couple.
        for k in antpos:
            Ak = ants[np.argwhere([np.all(np.isclose(k, ENU)) for ENU in antpos])[0][0]] #to get the antenna number from antenna ENU, 
            if np.linalg.norm(k-ant_i_enu ) !=0 : #k is not the ith antenna
                #Calculate azimuth KI-hat from ENU
                enu_ki = np.array(ant_i_enu-k)
                az_bl = np.arctan(enu_ki[1]/enu_ki[0])#radians
                
                pix_az_bl_at_horizon = hp.ang2pix(HH_beam.nside,za_bl, az_bl) 
                HH_at_bl = HH_beam.data_array[0,0,ipol,:,pix_az_bl_at_horizon] #one pixel, for all BEAM frequencies , if beam val: HH_cube[ii_az_bl_at_horizon, :]
                
                if same_beam_all_freqs:
                    ii_V_freqs_in_beam_array = np.array([105 for f_simulated in freq]).astype(int) 
                else:
                    ii_V_freqs_in_beam_array = np.array([find_nearest_index(HH_beam.freq_array[0], f_simulated) for f_simulated in freq]).astype(int) #assumes Hz, both freq arrays
                HH_at_bl = HH_at_bl[ii_V_freqs_in_beam_array] 

                #calculate zeroth order visibility, kj
                enu_kj = np.array(ant_j_enu-k)                
                Vkj0= uvd0.get_data((Ak,Aj, dict_coupling['pol']))# any redundant baseline with same ENU will work. BUT should it be (Ak,Aj) or (Aj, Ak) sticking with convention enu_kj= k->j // verified by uvw_array convention https://pyuvdata.readthedocs.io/en/latest/uvdata_parameters.html)


                #calculate ki terms
                propagator_dir= 1.
                if dict_coupling['alternate_coupling']['flip_propagator_dir']: propagator_dir= -1.
                coupling_ki = ((coupling_constants*gamma_Cheby)/np.linalg.norm(enu_ki))*exp( (propagator_dir*2. * np.pi * (np.linalg.norm(enu_ki)/(c_ms / freq))) )

                #Limit certain baselines, based on visibility term not delay term
                E = enu_kj[0]; N = enu_kj[1]; ABS = np.linalg.norm(enu_kj);
                if dict_coupling['limit_coupling']['type'] == 'thresh':
                    if eval(dict_coupling['limit_coupling']['E/W']) and eval(dict_coupling['limit_coupling']['N/S']) and eval(dict_coupling['limit_coupling']['ABS']): # execute thresholds from YAML
                        iants_knoti+=1
                        ants_coupled.append(Ak)
                        
                        coupling_sigma_knoti = coupling_sigma_knoti + coupling_ki*HH_at_bl*Vkj0
                elif dict_coupling['limit_coupling']['type'] == 'manual' and dict_coupling['IJK_manual_coupling'] == tuple((Ai,Aj,Ak)):
                    print('    In sigma_knoti, using manual IJK coupling for: '+str( tuple((Ai,Aj,Ak)) ))
                    iants_knoti+=1
                    ants_coupled.append(Ak)
                    coupling_sigma_knoti = coupling_sigma_knoti + coupling_ki*HH_at_bl*Vkj0
                elif dict_coupling['limit_coupling']['type'] == 'all':
                    iants_knoti+=1
                    ants_coupled.append(Ak)
                    coupling_sigma_knoti = coupling_sigma_knoti + coupling_ki*HH_at_bl*Vkj0

            if np.linalg.norm(k-ant_j_enu ) !=0 : #k is not the jth antenna
                #Calculate azimuth KJ-hat from ENU
                enu_kj = np.array(ant_j_enu-k)
                az_bl = np.arctan(enu_kj[1]/enu_kj[0]) #radians
                
                pix_az_bl_at_horizon = hp.ang2pix(HdagHdag_beam.nside,za_bl, az_bl) 
                HdagHdag_at_bl = HdagHdag_beam.data_array[0,0,ipol,:,pix_az_bl_at_horizon] #one pixel, for all BEAM frequencies , if beam val: #HdagHdag_cube[ii_az_bl_at_horizon, :] 
                
                if same_beam_all_freqs:
                    ii_V_freqs_in_beam_array = np.array([105 for f_simulated in freq]).astype(int) 
                else:
                    ii_V_freqs_in_beam_array = np.array([find_nearest_index(HdagHdag_beam.freq_array[0], f_simulated) for f_simulated in freq]).astype(int) #assumes Hz, both freq arrays
                HdagHdag_at_bl = HdagHdag_at_bl[ii_V_freqs_in_beam_array]  

                #calculate zeroth order visibility, ik
                enu_ik = np.array(k-ant_i_enu)
                Vik0= uvd0.get_data((Ai,Ak,dict_coupling['pol'])) 

                #calculate kj terms
                propagator_dir= -1.
                if dict_coupling['alternate_coupling']['flip_propagator_dir']: propagator_dir= 1.
                coupling_kj = ((coupling_constants*gamma_Cheby)/np.linalg.norm(enu_kj))*exp( (propagator_dir*2. * np.pi * (np.linalg.norm(enu_kj)/(c_ms / freq))) ) #yes, +2pi, not -2pi, in the exponent
                
                #Limit certain baselines, based on visibility term not delay term
                E = enu_ik[0]; N = enu_ik[1]; ABS = np.linalg.norm(enu_ik);
                if dict_coupling['limit_coupling']['type'] == 'thresh':

                    if eval(dict_coupling['limit_coupling']['E/W']) and eval(dict_coupling['limit_coupling']['N/S']) and eval(dict_coupling['limit_coupling']['ABS']): # execute thresholds from YAML
                        ants_coupled.append(Ak)
                        iants_knotj+=1
                        coupling_sigma_knotj = coupling_sigma_knotj + coupling_kj*Vik0*HdagHdag_at_bl
                elif dict_coupling['limit_coupling']['type'] == 'manual' and dict_coupling['IJK_manual_coupling'] == tuple((Ai,Aj,Ak)):
                    iants_knotj+=1
                    ants_coupled.append(Ak)
                    coupling_sigma_knotj = coupling_sigma_knotj + coupling_kj*Vik0*HdagHdag_at_bl
                elif dict_coupling['limit_coupling']['type'] == 'all':
                    iants_knotj+=1
                    ants_coupled.append(Ak)
                    coupling_sigma_knotj = coupling_sigma_knotj + coupling_kj*Vik0*HdagHdag_at_bl
        
    if dict_coupling['alternate_coupling']['sum_both_sigma']:
        Vij1 = Vij0 + coupling_sigma_knoti + coupling_sigma_knotj
    if dict_coupling['alternate_coupling']['subtr_both_sigma']:
        Vij1 = Vij0 - coupling_sigma_knoti - coupling_sigma_knotj
    elif dict_coupling['alternate_coupling']['flip_sigma_sum']:
        Vij1 = Vij0 + coupling_sigma_knoti - coupling_sigma_knotj
    else:
        Vij1 = Vij0 - coupling_sigma_knoti + coupling_sigma_knotj

    return antpairpol, Vij1

def calc_first_order_fringes_and_delays(uvd0, dict_coupling):
    """
    az, za = Azimuth, zenith angle, radians
    freq = frequeny in Hz
    enu = 'ij'th baseline vector in meters
    """
    DEC_HERA = -30.72113
    OMEGA_EARTH=1/(24*3600)

    d_fr_tau={}
    d_ants_coupled = {}

    pol = dict_coupling['pol']
    same_beam_all_freqs = dict_coupling['same_beam_all_freqs']
    antpos, ants = uvd0.get_ENU_antpos(pick_data_ants=False)
    print('    Looping over antpairpols...')
    for antpairpol in dict_coupling['keys_out']:

        antpairpol=tuple(antpairpol)
        d_fr_tau[antpairpol]={}
        ant_i_enu = antpos[np.nonzero(antpairpol[0]==ants)[0][0]]
        ant_j_enu = antpos[np.nonzero(antpairpol[1]==ants)[0][0]]
        Ai = ants[np.argwhere([np.all(np.isclose(ant_i_enu, ENU)) for ENU in antpos])[0][0]]
        Aj = ants[np.argwhere([np.all(np.isclose(ant_j_enu, ENU)) for ENU in antpos])[0][0]]
        enu= ant_j_enu - ant_i_enu
        baseline_number = uvd0.antnums_to_baseline(antpairpol[0],antpairpol[1])
        freq=uvd0.freq_array[0]

        print('    antpairpol (Vij0): '+str(antpairpol))

        #WARNING, TEMPORARY, SINCE NOT USING HEALVIS.BEAM_MODEL.BEAM_VAL for HH and HdagHdag
        ipol=[]
        if pol == 'XX' or pol =='xx': 
            ipol=0
        elif pol== 'YY' or pol =='yy': 
                ipol=1
        else:
            print('    ERROR when calculating first order visibility. invalid pol. only use diagonals for now, when using HH HdagHdag.')
            return
        
        za_bl = np.pi/2 #The zenith angle of all baselines is 90 degrees = pi/2
        ii_V_freqs_in_beam_array=[]

        iants_knoti=0
        iants_knotj=0
        ants_coupled = [] #keep track of all unique Ak that are used in coupling.
        for k in antpos:
            Ak = ants[np.argwhere([np.all(np.isclose(k, ENU)) for ENU in antpos])[0][0]] #to get the antenna number from antenna ENU, 
            d_fr_tau[antpairpol][Ak]={}
            if np.linalg.norm(k-ant_i_enu ) !=0 : #k is not the ith antenna
                #Calculate azimuth KI-hat from ENU
                enu_ki = np.array(ant_i_enu-k)
                az_bl = np.arctan(enu_ki[1]/enu_ki[0])#radians

                #calculate zeroth order visibility, kj
                enu_kj = np.array(ant_j_enu-k)
                fringe_kj_zenith = 1000.*OMEGA_EARTH*freq[0]*enu_kj[0]*np.cos(DEC_HERA*np.pi/180.)/c_ms #units [mHz]
                fringe_kj_horizon_galactic_set = 1000.*OMEGA_EARTH*freq[0]*np.sin(DEC_HERA*np.pi/180.)*enu_kj[1]/c_ms #units [mHz]

                #calculate ki terms
                propagator_dir= 1.
                if dict_coupling['alternate_coupling']['flip_propagator_dir']: propagator_dir= -1.
                tau_ki = -np.linalg.norm(enu_ki)/0.3 #opposite of the formalism, to comapre to uvtools delay space plot. numpy fft uses exp(-2pi i ) convention, not exp(+2pi i)

                #Limit certain baselines, based on visibility term not delay term
                E = enu_kj[0]; N = enu_kj[1]; ABS = np.linalg.norm(enu_kj);
                if dict_coupling['limit_coupling']['type'] == 'thresh':
                    if eval(dict_coupling['limit_coupling']['E/W']) and eval(dict_coupling['limit_coupling']['N/S']) and eval(dict_coupling['limit_coupling']['ABS']): # execute thresholds from YAML
                        iants_knoti+=1
                        ants_coupled.append(Ak)
                        d_fr_tau[antpairpol][Ak]['tau_k!=i']=tau_ki
                        d_fr_tau[antpairpol][Ak]['f_h_k!=i']=fringe_kj_horizon_galactic_set
                        d_fr_tau[antpairpol][Ak]['f_z_k!=i']=fringe_kj_zenith

                elif dict_coupling['limit_coupling']['type'] == 'manual' and dict_coupling['IJK_manual_coupling'] == tuple((Ai,Aj,Ak)):
                    iants_knoti+=1
                    ants_coupled.append(Ak)
                    d_fr_tau[antpairpol][Ak]['tau_k!=i']=tau_ki
                    d_fr_tau[antpairpol][Ak]['f_h_k!=i']=fringe_kj_horizon_galactic_set
                    d_fr_tau[antpairpol][Ak]['f_z_k!=i']=fringe_kj_zenith
                elif dict_coupling['limit_coupling']['type'] == 'all':
                    iants_knoti+=1
                    ants_coupled.append(Ak)
                    d_fr_tau[antpairpol][Ak]['tau_k!=i']=tau_ki
                    d_fr_tau[antpairpol][Ak]['f_h_k!=i']=fringe_kj_horizon_galactic_set
                    d_fr_tau[antpairpol][Ak]['f_z_k!=i']=fringe_kj_zenith

            if np.linalg.norm(k-ant_j_enu ) !=0 : #k is not the jth antenna
                #Calculate azimuth KJ-hat from ENU
                enu_kj = np.array(ant_j_enu-k)
                az_bl = np.arctan(enu_kj[1]/enu_kj[0]) #radians
                
                #calculate zeroth order visibility, ik
                enu_ik = np.array(k-ant_i_enu)
                fringe_ik_zenith = 1000.*OMEGA_EARTH*freq[0]*enu_ik[0]*np.cos(DEC_HERA*np.pi/180.)/c_ms  #units [mHz]
                fringe_ik_horizon_galactic_set = 1000.*OMEGA_EARTH*freq[0]*np.sin(DEC_HERA*np.pi/180.)*enu_ik[1]/c_ms #units [mHz]


                #calculate kj terms
                propagator_dir= -1.
                if dict_coupling['alternate_coupling']['flip_propagator_dir']: propagator_dir= 1.
                tau_kj = 1.*np.linalg.norm(enu_kj)/0.3
                
                #Limit certain baselines, based on visibility term not delay term
                E = enu_ik[0]; N = enu_ik[1]; ABS = np.linalg.norm(enu_ik);
                if dict_coupling['limit_coupling']['type'] == 'thresh':
                    #print('    type of dict_coupling[limit_coupling][E/W]: '+str(type(dict_coupling['limit_coupling']['E/W'])))
                    #print('    eval statement values: '+str( eval(dict_coupling['limit_coupling']['E/W']) )+', '+str(eval(dict_coupling['limit_coupling']['N/S']))+', '+str(eval(dict_coupling['limit_coupling']['ABS'])))
                    if eval(dict_coupling['limit_coupling']['E/W']) and eval(dict_coupling['limit_coupling']['N/S']) and eval(dict_coupling['limit_coupling']['ABS']): # execute thresholds from YAML
                        #print('      In sigma_knotj,IJK = : '+str( tuple((Ai,Aj,Ak)) ) +' with ENU_ik: '+str(np.round(enu_ik,1)))
                        iants_knotj+=1
                        ants_coupled.append(Ak)
                        d_fr_tau[antpairpol][Ak]['tau_k!=j']=tau_kj
                        d_fr_tau[antpairpol][Ak]['f_h_k!=j']=fringe_ik_horizon_galactic_set
                        d_fr_tau[antpairpol][Ak]['f_z_k!=j']=fringe_ik_zenith
                elif dict_coupling['limit_coupling']['type'] == 'manual' and dict_coupling['IJK_manual_coupling'] == tuple((Ai,Aj,Ak)): #NOT SURE THIS WORKS, MUST CHECK
                    #print('    In sigma_knotj, using manual IJK coupling for: '+str( tuple((Ai,Aj,Ak)) ))
                    iants_knotj+=1
                    ants_coupled.append(Ak)
                    d_fr_tau[antpairpol][Ak]['tau_k!=j']=tau_kj
                    d_fr_tau[antpairpol][Ak]['f_h_k!=j']=fringe_ik_horizon_galactic_set
                    d_fr_tau[antpairpol][Ak]['f_z_k!=j']=fringe_ik_zenith
                elif dict_coupling['limit_coupling']['type'] == 'all':
                    iants_knotj+=1
                    ants_coupled.append(Ak)
                    d_fr_tau[antpairpol][Ak]['tau_k!=j']=tau_kj
                    d_fr_tau[antpairpol][Ak]['f_h_k!=j']=fringe_ik_horizon_galactic_set
                    d_fr_tau[antpairpol][Ak]['f_z_k!=j']=fringe_ik_zenith
                elif dict_coupling['limit_coupling']['type'] == 'none': #do not couple anything
                    continue
        
        d_ants_coupled[antpairpol]=ants_coupled
    print('    Done.')
    print('    In last loop over ants, Ants looped over for k!=i : '+str(iants_knoti)+', k!=j : '+str(iants_knotj))
    return d_fr_tau, d_ants_coupled
