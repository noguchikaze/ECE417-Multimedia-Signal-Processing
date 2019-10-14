import matplotlib.pyplot as plt
import numpy as np
from submitted import Spectrograph

# Save a spectrogram image as a PNG file
#    Assumes that you have already run Spectrograph.set_timeaxis and Spectrograph.set_image
def save_spectrogram_image(spectrograph, img_filename):
    maxtime = max(spectrograph.timeaxis)
    maxfreq = spectrograph.maxfreq
    image = spectrograph.image
    plt.imshow(np.transpose(image),origin='lower',extent=(0,maxtime,0,maxfreq),aspect='auto')
    plt.xlabel('Time (sec)')
    plt.ylabel('Freq (Hz)')
    plt.savefig(img_filename)

        
        
    
