#!/usr/bin/python3

"""
    ============================================
    = Galactocentric Properties from Pace 2024 =
    ============================================

    Gets all of the galactocentric properties and the errors from the
    Pace 2024 paper.
"""

# Import packages
import corner
import numpy as np
import matplotlib.pyplot as plt
import astropy.table as table
from astropy import units as u
import astropy.coordinates as coord
#
coord.galactocentric_frame_defaults.set('v4.0')
gc_frame = coord.Galactocentric()

# Get the table from Github
dsph_mw = table.Table.read('https://raw.githubusercontent.com/apace7/local_volume_database/main/data/dwarf_mw.csv')

# Add astropy units to the data
c_dsph_mw = coord.SkyCoord(ra=dsph_mw['ra']*u.deg, dec=dsph_mw['dec']*u.deg,  frame='icrs', distance=dsph_mw['distance']*u.kpc, radial_velocity=dsph_mw['vlos_systemic']*u.km/u.s, pm_ra_cosdec=dsph_mw['pmra']*u.mas/u.yr, pm_dec=dsph_mw['pmdec']*u.mas/u.yr)

# Initialize a bunch of empty arrays
dsph_mw['vrad'] = np.zeros(len(dsph_mw), dtype=float)
dsph_mw['vtan'] = np.zeros(len(dsph_mw), dtype=float)
dsph_mw['vrad_error'] = np.zeros(len(dsph_mw), dtype=float)
dsph_mw['vtan_error'] = np.zeros(len(dsph_mw), dtype=float)
#
dsph_mw['distance_gc'] = np.zeros(len(dsph_mw), dtype=float)
dsph_mw['distance_gc_em'] = np.zeros(len(dsph_mw), dtype=float)
dsph_mw['distance_gc_ep'] = np.zeros(len(dsph_mw), dtype=float)
dsph_mw['distance_gc_68'] = np.zeros(len(dsph_mw), dtype=float)

# Loop over all of the MW satellites
for i in range(len(dsph_mw)):
    #
    # Set astropy coordinates for each of the satellites
    icrs2 = coord.SkyCoord(ra=dsph_mw['ra'][i]*u.deg,
                          dec=dsph_mw['dec'][i]*u.deg,
                          distance=dsph_mw['distance'][i]*u.kpc,
                          pm_ra_cosdec=dsph_mw['pmra'][i]*u.mas/u.yr,
                          pm_dec=dsph_mw['pmdec'][i]*u.mas/u.yr,
                          radial_velocity=dsph_mw['vlos_systemic'][i]*u.km/u.s)
    #
    # Set the average error for a satellite in astropy coordinates
    icrs_err2 = coord.SkyCoord(ra=0*u.deg, dec=0*u.deg, distance=(dsph_mw['distance_em'][i]+dsph_mw['distance_ep'][i])/2.*u.kpc,
                              pm_ra_cosdec=(dsph_mw['pmra_em'][i]+dsph_mw['pmra_ep'][i])/2.*u.mas/u.yr,
                              pm_dec=(dsph_mw['pmdec_em'][i]+dsph_mw['pmdec_ep'][i])/2.*u.mas/u.yr,
                              radial_velocity=(dsph_mw['vlos_systemic_em'][i]+dsph_mw['vlos_systemic_ep'][i])/2.*u.km/u.s)
    #
    # For each coordinate property, sample from a Normal distribution 1000x
    n_samples = 1000
    #
    dist = np.random.normal(icrs2.distance.value, icrs_err2.distance.value,
                            n_samples) * icrs2.distance.unit
    #
    pm_ra_cosdec = np.random.normal(icrs2.pm_ra_cosdec.value,
                                    icrs_err2.pm_ra_cosdec.value,
                                    n_samples) * icrs2.pm_ra_cosdec.unit
    #
    pm_dec = np.random.normal(icrs2.pm_dec.value,
                              icrs_err2.pm_dec.value,
                              n_samples) * icrs2.pm_dec.unit
    #
    rv = np.random.normal(icrs2.radial_velocity.value, icrs_err2.radial_velocity.value,
                          n_samples) * icrs2.radial_velocity.unit
    #
    ra = np.full(n_samples, icrs2.ra.degree) * u.degree
    dec = np.full(n_samples, icrs2.dec.degree) * u.degree
    #
    # Put all of the samples together into one object
    icrs_samples = coord.SkyCoord(ra=ra, dec=dec, distance=dist,
                              pm_ra_cosdec=pm_ra_cosdec,
                              pm_dec=pm_dec, radial_velocity=rv)
    
    # Convert from the ICRS Frame to the Galactocentric Frame
    galcen_samples = icrs_samples.transform_to(gc_frame)
    
    # Calculate the total distances, radial velocities, and the tangential velocities
    r = np.sqrt(galcen_samples.x**2 + galcen_samples.y**2 + galcen_samples.z**2)
    #
    vrad = galcen_samples.x*galcen_samples.v_x/r + galcen_samples.y*galcen_samples.v_y/r + galcen_samples.z*galcen_samples.v_z/r
    #
    vtan = np.sqrt((r*galcen_samples.v_z - galcen_samples.z *vrad)**2+(galcen_samples.x*galcen_samples.v_y - galcen_samples.y*galcen_samples.v_x)**2) \
    /np.sqrt(galcen_samples.x**2 + galcen_samples.y**2)

    # Get the median, and the width of the 68th percentile as the error
    pers_quant = corner.quantile(vrad.value, [.5, .1587, .8413], )
    dsph_mw['vrad'][i] = pers_quant[0]
    dsph_mw['vrad_error'][i] = (pers_quant[2]-pers_quant[1])/2.
    #
    pers_quant = corner.quantile(vtan.value, [.5, .1587, .8413], )
    dsph_mw['vtan'][i] = pers_quant[0]
    dsph_mw['vtan_error'][i] = (pers_quant[2]-pers_quant[1])/2.
    #
    pers_quant = corner.quantile(vtan.value, [.5, .1587, .8413], )
    dsph_mw['vtan'][i] = pers_quant[0]
    dsph_mw['vtan_error'][i] = (pers_quant[2]-pers_quant[1])/2.
    #    
    pers_quant = corner.quantile(np.sqrt(galcen_samples.x.value**2 + galcen_samples.y.value**2 + galcen_samples.z.value**2), [.5, .1587, .8413], )
    dsph_mw['distance_gc'][i] = pers_quant[0]
    dsph_mw['distance_gc_ep'][i] = (pers_quant[2]-pers_quant[0])
    dsph_mw['distance_gc_em'][i] = (pers_quant[0]-pers_quant[1])
    dsph_mw['distance_gc_68'][i] = (pers_quant[2]-pers_quant[1])/2


# Get the table from Github
gcs_amb = table.Table.read('https://raw.githubusercontent.com/apace7/local_volume_database/main/data/gc_ambiguous.csv')

# Add astropy units to the data
c_gcs_amb = coord.SkyCoord(ra=gcs_amb['ra']*u.deg, dec=gcs_amb['dec']*u.deg,  frame='icrs', distance=gcs_amb['distance']*u.kpc, radial_velocity=gcs_amb['vlos_systemic']*u.km/u.s, pm_ra_cosdec=gcs_amb['pmra']*u.mas/u.yr,pm_dec=gcs_amb['pmdec']*u.mas/u.yr)

# Initialize a bunch of empty arrays
gcs_amb['vrad'] = np.zeros(len(gcs_amb), dtype=float)
gcs_amb['vtan'] = np.zeros(len(gcs_amb), dtype=float)
gcs_amb['vrad_error'] = np.zeros(len(gcs_amb), dtype=float)
gcs_amb['vtan_error'] = np.zeros(len(gcs_amb), dtype=float)
#
gcs_amb['distance_gc'] = np.zeros(len(gcs_amb), dtype=float)
gcs_amb['distance_gc_em'] = np.zeros(len(gcs_amb), dtype=float)
gcs_amb['distance_gc_ep'] = np.zeros(len(gcs_amb), dtype=float)
gcs_amb['distance_gc_68'] = np.zeros(len(gcs_amb), dtype=float)

# Loop over all of the MW satellites
for i in range(len(gcs_amb)):
    #
    # Set astropy coordinates for each of the satellites
    icrs2 = coord.SkyCoord(ra=gcs_amb['ra'][i]*u.deg,
                          dec=gcs_amb['dec'][i]*u.deg,
                          distance=gcs_amb['distance'][i]*u.kpc,
                          pm_ra_cosdec=gcs_amb['pmra'][i]*u.mas/u.yr,
                          pm_dec=gcs_amb['pmdec'][i]*u.mas/u.yr,
                          radial_velocity=gcs_amb['vlos_systemic'][i]*u.km/u.s)
    #
    # Set the average error for a satellite in astropy coordinates
    icrs_err2 = coord.SkyCoord(ra=0*u.deg, dec=0*u.deg, distance=(gcs_amb['distance_em'][i]+gcs_amb['distance_ep'][i])/2.*u.kpc,
                              pm_ra_cosdec=(gcs_amb['pmra_em'][i]+gcs_amb['pmra_ep'][i])/2.*u.mas/u.yr,
                              pm_dec=(gcs_amb['pmdec_em'][i]+gcs_amb['pmdec_ep'][i])/2.*u.mas/u.yr,
                              radial_velocity=(gcs_amb['vlos_systemic_em'][i]+gcs_amb['vlos_systemic_ep'][i])/2.*u.km/u.s)
    #
    # For each coordinate property, sample from a Normal distribution 1000x
    n_samples = 1000
    #
    dist = np.random.normal(icrs2.distance.value, icrs_err2.distance.value,
                            n_samples) * icrs2.distance.unit
    #
    pm_ra_cosdec = np.random.normal(icrs2.pm_ra_cosdec.value,
                                    icrs_err2.pm_ra_cosdec.value,
                                    n_samples) * icrs2.pm_ra_cosdec.unit
    #
    pm_dec = np.random.normal(icrs2.pm_dec.value,
                              icrs_err2.pm_dec.value,
                              n_samples) * icrs2.pm_dec.unit
    #
    rv = np.random.normal(icrs2.radial_velocity.value, icrs_err2.radial_velocity.value,
                          n_samples) * icrs2.radial_velocity.unit
    #
    ra = np.full(n_samples, icrs2.ra.degree) * u.degree
    dec = np.full(n_samples, icrs2.dec.degree) * u.degree
    #
    # Put all of the samples together into one object
    icrs_samples = coord.SkyCoord(ra=ra, dec=dec, distance=dist,
                              pm_ra_cosdec=pm_ra_cosdec,
                              pm_dec=pm_dec, radial_velocity=rv)
    
    # Convert from the ICRS Frame to the Galactocentric Frame
    galcen_samples = icrs_samples.transform_to(gc_frame)
    
    # Calculate the total distances, radial velocities, and the tangential velocities
    r = np.sqrt(galcen_samples.x**2 + galcen_samples.y**2 + galcen_samples.z**2)
    #
    vrad = galcen_samples.x*galcen_samples.v_x/r + galcen_samples.y*galcen_samples.v_y/r + galcen_samples.z*galcen_samples.v_z/r
    #
    vtan = np.sqrt((r*galcen_samples.v_z - galcen_samples.z *vrad)**2+(galcen_samples.x*galcen_samples.v_y - galcen_samples.y*galcen_samples.v_x)**2) \
    /np.sqrt(galcen_samples.x**2 + galcen_samples.y**2)

    # Get the median, and the width of the 68th percentile as the error
    pers_quant = corner.quantile(vrad.value, [.5, .1587, .8413], )
    gcs_amb['vrad'][i] = pers_quant[0]
    gcs_amb['vrad_error'][i] = (pers_quant[2]-pers_quant[1])/2.
    #
    pers_quant = corner.quantile(vtan.value, [.5, .1587, .8413], )
    gcs_amb['vtan'][i] = pers_quant[0]
    gcs_amb['vtan_error'][i] = (pers_quant[2]-pers_quant[1])/2.
    #
    pers_quant = corner.quantile(vtan.value, [.5, .1587, .8413], )
    gcs_amb['vtan'][i] = pers_quant[0]
    gcs_amb['vtan_error'][i] = (pers_quant[2]-pers_quant[1])/2.
    #    
    pers_quant = corner.quantile(np.sqrt(galcen_samples.x.value**2 + galcen_samples.y.value**2 + galcen_samples.z.value**2), [.5, .1587, .8413], )
    gcs_amb['distance_gc'][i] = pers_quant[0]
    gcs_amb['distance_gc_ep'][i] = (pers_quant[2]-pers_quant[0])
    gcs_amb['distance_gc_em'][i] = (pers_quant[0]-pers_quant[1])
    gcs_amb['distance_gc_68'][i] = (pers_quant[2]-pers_quant[1])/2


"""
    ALL dwarfs in Pace data
"""
# Get the table from Github
dwf_all = table.Table.read('https://raw.githubusercontent.com/apace7/local_volume_database/main/data/dwarf_all.csv')

# Add astropy units to the data
c_dwf_all = coord.SkyCoord(ra=dwf_all['ra']*u.deg, dec=dwf_all['dec']*u.deg,  frame='icrs', distance=dwf_all['distance']*u.kpc, radial_velocity=dwf_all['vlos_systemic']*u.km/u.s, pm_ra_cosdec=dwf_all['pmra']*u.mas/u.yr,pm_dec=dwf_all['pmdec']*u.mas/u.yr)

# Initialize a bunch of empty arrays
dwf_all['vrad'] = np.zeros(len(dwf_all), dtype=float)
dwf_all['vtan'] = np.zeros(len(dwf_all), dtype=float)
dwf_all['vrad_error'] = np.zeros(len(dwf_all), dtype=float)
dwf_all['vtan_error'] = np.zeros(len(dwf_all), dtype=float)
#
dwf_all['distance_gc'] = np.zeros(len(dwf_all), dtype=float)
dwf_all['distance_gc_em'] = np.zeros(len(dwf_all), dtype=float)
dwf_all['distance_gc_ep'] = np.zeros(len(dwf_all), dtype=float)
dwf_all['distance_gc_68'] = np.zeros(len(dwf_all), dtype=float)

# Loop over all of the MW satellites
for i in range(len(dwf_all)):
    #
    # Set astropy coordinates for each of the satellites
    icrs2 = coord.SkyCoord(ra=dwf_all['ra'][i]*u.deg,
                          dec=dwf_all['dec'][i]*u.deg,
                          distance=dwf_all['distance'][i]*u.kpc,
                          pm_ra_cosdec=dwf_all['pmra'][i]*u.mas/u.yr,
                          pm_dec=dwf_all['pmdec'][i]*u.mas/u.yr,
                          radial_velocity=dwf_all['vlos_systemic'][i]*u.km/u.s)
    #
    # Set the average error for a satellite in astropy coordinates
    icrs_err2 = coord.SkyCoord(ra=0*u.deg, dec=0*u.deg, distance=(dwf_all['distance_em'][i]+dwf_all['distance_ep'][i])/2.*u.kpc,
                              pm_ra_cosdec=(dwf_all['pmra_em'][i]+dwf_all['pmra_ep'][i])/2.*u.mas/u.yr,
                              pm_dec=(dwf_all['pmdec_em'][i]+dwf_all['pmdec_ep'][i])/2.*u.mas/u.yr,
                              radial_velocity=(dwf_all['vlos_systemic_em'][i]+dwf_all['vlos_systemic_ep'][i])/2.*u.km/u.s)
    #
    # For each coordinate property, sample from a Normal distribution 1000x
    n_samples = 1000
    #
    dist = np.random.normal(icrs2.distance.value, icrs_err2.distance.value,
                            n_samples) * icrs2.distance.unit
    #
    pm_ra_cosdec = np.random.normal(icrs2.pm_ra_cosdec.value,
                                    icrs_err2.pm_ra_cosdec.value,
                                    n_samples) * icrs2.pm_ra_cosdec.unit
    #
    pm_dec = np.random.normal(icrs2.pm_dec.value,
                              icrs_err2.pm_dec.value,
                              n_samples) * icrs2.pm_dec.unit
    #
    rv = np.random.normal(icrs2.radial_velocity.value, icrs_err2.radial_velocity.value,
                          n_samples) * icrs2.radial_velocity.unit
    #
    ra = np.full(n_samples, icrs2.ra.degree) * u.degree
    dec = np.full(n_samples, icrs2.dec.degree) * u.degree
    #
    # Put all of the samples together into one object
    icrs_samples = coord.SkyCoord(ra=ra, dec=dec, distance=dist,
                              pm_ra_cosdec=pm_ra_cosdec,
                              pm_dec=pm_dec, radial_velocity=rv)
    
    # Convert from the ICRS Frame to the Galactocentric Frame
    galcen_samples = icrs_samples.transform_to(gc_frame)
    
    # Calculate the total distances, radial velocities, and the tangential velocities
    r = np.sqrt(galcen_samples.x**2 + galcen_samples.y**2 + galcen_samples.z**2)
    #
    vrad = galcen_samples.x*galcen_samples.v_x/r + galcen_samples.y*galcen_samples.v_y/r + galcen_samples.z*galcen_samples.v_z/r
    #
    vtan = np.sqrt((r*galcen_samples.v_z - galcen_samples.z *vrad)**2+(galcen_samples.x*galcen_samples.v_y - galcen_samples.y*galcen_samples.v_x)**2) \
    /np.sqrt(galcen_samples.x**2 + galcen_samples.y**2)

    # Get the median, and the width of the 68th percentile as the error
    pers_quant = corner.quantile(vrad.value, [.5, .1587, .8413], )
    dwf_all['vrad'][i] = pers_quant[0]
    dwf_all['vrad_error'][i] = (pers_quant[2]-pers_quant[1])/2.
    #
    pers_quant = corner.quantile(vtan.value, [.5, .1587, .8413], )
    dwf_all['vtan'][i] = pers_quant[0]
    dwf_all['vtan_error'][i] = (pers_quant[2]-pers_quant[1])/2.
    #
    pers_quant = corner.quantile(vtan.value, [.5, .1587, .8413], )
    dwf_all['vtan'][i] = pers_quant[0]
    dwf_all['vtan_error'][i] = (pers_quant[2]-pers_quant[1])/2.
    #    
    pers_quant = corner.quantile(np.sqrt(galcen_samples.x.value**2 + galcen_samples.y.value**2 + galcen_samples.z.value**2), [.5, .1587, .8413], )
    dwf_all['distance_gc'][i] = pers_quant[0]
    dwf_all['distance_gc_ep'][i] = (pers_quant[2]-pers_quant[0])
    dwf_all['distance_gc_em'][i] = (pers_quant[0]-pers_quant[1])
    dwf_all['distance_gc_68'][i] = (pers_quant[2]-pers_quant[1])/2





"""
    ALL dwarfs in Pace data
"""
# Get the table from Github
gcs_new = table.Table.read('https://raw.githubusercontent.com/apace7/local_volume_database/main/data/gc_mw_new.csv')

# Add astropy units to the data
c_gcs_new = coord.SkyCoord(ra=gcs_new['ra']*u.deg, dec=gcs_new['dec']*u.deg,  frame='icrs', distance=gcs_new['distance']*u.kpc, radial_velocity=gcs_new['vlos_systemic']*u.km/u.s, pm_ra_cosdec=gcs_new['pmra']*u.mas/u.yr,pm_dec=gcs_new['pmdec']*u.mas/u.yr)

# Initialize a bunch of empty arrays
gcs_new['vrad'] = np.zeros(len(gcs_new), dtype=float)
gcs_new['vtan'] = np.zeros(len(gcs_new), dtype=float)
gcs_new['vrad_error'] = np.zeros(len(gcs_new), dtype=float)
gcs_new['vtan_error'] = np.zeros(len(gcs_new), dtype=float)
#
gcs_new['distance_gc'] = np.zeros(len(gcs_new), dtype=float)
gcs_new['distance_gc_em'] = np.zeros(len(gcs_new), dtype=float)
gcs_new['distance_gc_ep'] = np.zeros(len(gcs_new), dtype=float)
gcs_new['distance_gc_68'] = np.zeros(len(gcs_new), dtype=float)

# Loop over all of the MW satellites
for i in range(len(gcs_new)):
    #
    # Set astropy coordinates for each of the satellites
    icrs2 = coord.SkyCoord(ra=gcs_new['ra'][i]*u.deg,
                          dec=gcs_new['dec'][i]*u.deg,
                          distance=gcs_new['distance'][i]*u.kpc,
                          pm_ra_cosdec=gcs_new['pmra'][i]*u.mas/u.yr,
                          pm_dec=gcs_new['pmdec'][i]*u.mas/u.yr,
                          radial_velocity=gcs_new['vlos_systemic'][i]*u.km/u.s)
    #
    # Set the average error for a satellite in astropy coordinates
    icrs_err2 = coord.SkyCoord(ra=0*u.deg, dec=0*u.deg, distance=(gcs_new['distance_em'][i]+gcs_new['distance_ep'][i])/2.*u.kpc,
                              pm_ra_cosdec=(gcs_new['pmra_em'][i]+gcs_new['pmra_ep'][i])/2.*u.mas/u.yr,
                              pm_dec=(gcs_new['pmdec_em'][i]+gcs_new['pmdec_ep'][i])/2.*u.mas/u.yr,
                              radial_velocity=(gcs_new['vlos_systemic_em'][i]+gcs_new['vlos_systemic_ep'][i])/2.*u.km/u.s)
    #
    # For each coordinate property, sample from a Normal distribution 1000x
    n_samples = 1000
    #
    dist = np.random.normal(icrs2.distance.value, icrs_err2.distance.value,
                            n_samples) * icrs2.distance.unit
    #
    pm_ra_cosdec = np.random.normal(icrs2.pm_ra_cosdec.value,
                                    icrs_err2.pm_ra_cosdec.value,
                                    n_samples) * icrs2.pm_ra_cosdec.unit
    #
    pm_dec = np.random.normal(icrs2.pm_dec.value,
                              icrs_err2.pm_dec.value,
                              n_samples) * icrs2.pm_dec.unit
    #
    rv = np.random.normal(icrs2.radial_velocity.value, icrs_err2.radial_velocity.value,
                          n_samples) * icrs2.radial_velocity.unit
    #
    ra = np.full(n_samples, icrs2.ra.degree) * u.degree
    dec = np.full(n_samples, icrs2.dec.degree) * u.degree
    #
    # Put all of the samples together into one object
    icrs_samples = coord.SkyCoord(ra=ra, dec=dec, distance=dist,
                              pm_ra_cosdec=pm_ra_cosdec,
                              pm_dec=pm_dec, radial_velocity=rv)
    
    # Convert from the ICRS Frame to the Galactocentric Frame
    galcen_samples = icrs_samples.transform_to(gc_frame)
    
    # Calculate the total distances, radial velocities, and the tangential velocities
    r = np.sqrt(galcen_samples.x**2 + galcen_samples.y**2 + galcen_samples.z**2)
    #
    vrad = galcen_samples.x*galcen_samples.v_x/r + galcen_samples.y*galcen_samples.v_y/r + galcen_samples.z*galcen_samples.v_z/r
    #
    vtan = np.sqrt((r*galcen_samples.v_z - galcen_samples.z *vrad)**2+(galcen_samples.x*galcen_samples.v_y - galcen_samples.y*galcen_samples.v_x)**2) \
    /np.sqrt(galcen_samples.x**2 + galcen_samples.y**2)

    # Get the median, and the width of the 68th percentile as the error
    pers_quant = corner.quantile(vrad.value, [.5, .1587, .8413], )
    gcs_new['vrad'][i] = pers_quant[0]
    gcs_new['vrad_error'][i] = (pers_quant[2]-pers_quant[1])/2.
    #
    pers_quant = corner.quantile(vtan.value, [.5, .1587, .8413], )
    gcs_new['vtan'][i] = pers_quant[0]
    gcs_new['vtan_error'][i] = (pers_quant[2]-pers_quant[1])/2.
    #
    pers_quant = corner.quantile(vtan.value, [.5, .1587, .8413], )
    gcs_new['vtan'][i] = pers_quant[0]
    gcs_new['vtan_error'][i] = (pers_quant[2]-pers_quant[1])/2.
    #    
    pers_quant = corner.quantile(np.sqrt(galcen_samples.x.value**2 + galcen_samples.y.value**2 + galcen_samples.z.value**2), [.5, .1587, .8413], )
    gcs_new['distance_gc'][i] = pers_quant[0]
    gcs_new['distance_gc_ep'][i] = (pers_quant[2]-pers_quant[0])
    gcs_new['distance_gc_em'][i] = (pers_quant[0]-pers_quant[1])
    gcs_new['distance_gc_68'][i] = (pers_quant[2]-pers_quant[1])/2




