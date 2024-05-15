import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits 
import astropy.convolution as apconv
import sys 
import muram as mio
import matplotlib
from tqdm import tqdm


# This convolves a series of 2D slices with the desired 2d function
# using astropy functionallity. 2D slices in this example are binary files 
# created by MURaM code

# Steps are : 
# 1) Prepare the PSF, convolve some test intensity data -> you can comment this out if 
#                                                          you don't have intensity

# 2) Loop through the files, read desired 2D quantity convolve, bin, store into a cube

# 3) Add needed header information, save cube into a fits. 

# 4) Repeat as needed for other quantities, making sure to change what is read and 
#    how you name the output file

# ------------------------------------------------------------------------------------------
# Prepare the PSF:
D = 1.0 #m 
llambda = 500E-9 # wavelengh in m 

pix_size = 16E3 # in m

noise = 0.0

pixel_scale_arcseconds = 16E3 / 725E3

diff_limit_arseconds = 1.22 * llambda / D * 206265

diff_limit_pixels = diff_limit_arseconds / pixel_scale_arcseconds

print("info::first zero of your PSF is at: ", diff_limit_pixels)

PSF = apconv.AiryDisk2DKernel(diff_limit_pixels, mode='oversample')

plt.figure(figsize=[6,5])
plt.imshow(PSF)
plt.savefig("psf.png",bbox_inches='tight')
plt.close('all')

# ------------------------------------------------------------------------------------------


path = '/mnt/c/Users/ivanz/OneDrive/Documents/Muram_ISSI_2D'
iter = 00000

test=mio.MuramIntensity(path, iter)

testc = apconv.convolve_fft(test,PSF,boundary='wrap',normalize_kernel=True)

qs = np.mean(test)
test /= qs
testc /= qs

testcb = np.sum(testc.reshape(768,2,768,2),axis=(1,3))/4.0

print (np.std(test))
print (np.std(testc))
print (np.std(testcb))
plt.figure(figsize=[19,5])

plt.subplot(131)
plt.imshow(test.T, origin='lower', cmap='magma',vmin=0.7,vmax=1.3)
plt.colorbar()

plt.subplot(132)
plt.imshow(testc.T, origin='lower', cmap='magma',vmin=0.7,vmax=1.3)
plt.colorbar()

plt.subplot(133)
plt.imshow(testcb.T, origin='lower', cmap='magma',vmin=0.7,vmax=1.3)
plt.colorbar()
 
plt.tight_layout()
plt.savefig("test_conv.png",bbox_inches='tight')

# -----------------------------------------------------------------------------

start = 0 #iter to start from 
step = 50 # step
number = 361 # total number 

for i in tqdm(range(number)):

	snapshot = start + i*step

	# For the intensity
	test=mio.MuramIntensity(path, snapshot)

	# For Bz (or anything else, we have to approach it differently:)
	#data = mio.read_slice(path, snapshot, 'tau', '1.000')
	#test = np.copy(data[0][5,:,:] * np.sqrt(4 * np.pi))
	#data = None
	
	# Convolve using astropy's convolve_fft
	testc = apconv.convolve_fft(test,PSF,boundary='wrap',normalize_kernel=True)
	# Bin
	testcb = np.sum(testc.reshape(768,2,768,2),axis=(1,3))/4.0

	if (i==0):
		cube = testcb.reshape(1,768,768)
	else:
		cube = np.append(cube,testcb[None,:,:],axis=0)

	test = None 
	testc = None  
	testcb = None 

print(cube.shape)

kek = fits.PrimaryHDU(cube)

# ------------------------------------------
# Prepare fits header:
kek.header['AXIS1'] = 'TIME [s]'
kek.header['AXIS2'] = 'X on solar surface [km]'
kek.header['AXIS3'] = 'Y on solar surface [km]'
kek.header['VALUES'] = 'Intensity [erg / cm^2 / s / srad / nm]'
kek.header['CONTEXT'] = 'I_c at 500 nm'
kek.header['DELTAT'] = 10
kek.header['DELTAX'] = 32
kek.header['DELTAY'] = 32
kek.header['DGRD'] = 'Convolved with Airy disk of 1m telescope at 500 nm'
kek.header['DATE'] = '15/05/2024'
kek.header['PLACE'] = 'ISSI meeting on flows'

kek.writeto('SSD_25_8Mm_16_pdmp_1_I_500.fits',overwrite=True)


