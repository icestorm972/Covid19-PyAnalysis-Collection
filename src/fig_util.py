# -*- coding: utf-8 -*-
import os
import tempfile
import numpy as np
from matplotlib.image import imread

    


def force_fig_size(fig, target_size_in_px, dpi=100, pad_inches=0.2, max_iter=8):
    
    def _win_named_temp_file(suffix):
        fname = os.path.join(tempfile.gettempdir(), os.urandom(24).hex() + suffix)
        return open(fname, "w") 

    def get_new_scale():
        with _win_named_temp_file('.png') as f:
            fig.savefig(f.name, bbox_inches='tight', dpi=dpi, pad_inches=pad_inches)
            height, width, _channels = imread(f.name).shape
            
            new_xscale = target_size_in_px[0] / width
            new_yscale = target_size_in_px[1] / height
                
            dx = width - target_size_in_px[0]
            dy = height - target_size_in_px[1]
            derr = int(np.ceil(abs(dx))) + int(np.ceil(abs(dy)))
            
            print('force_fig_size: dx = {:.2f} px   dy = {:.2f} px   delta = {:d} px'.format(dx, dy, derr))
            
            if (height < 20):
                print('WARNING: HEIGHT TOO SMALL! ABORT!')
            if (width < 20):
                print('WARNING: WIDTH TOO SMALL! ABORT!')
                
            return np.array([new_xscale, new_yscale]), derr, (height<20) or (width<20)
    
    new_size = np.asarray(target_size_in_px) / (1.0 * dpi)
    
    min_derr = 1.0e9
    num_diverg = 0
    while True:
        fig.set_size_inches(new_size)
        
        new_scale, derr, degener = get_new_scale()
        
        if degener:
            return False
        
        new_size *= new_scale
        
        if min_derr > derr:
            min_derr = derr
            num_diverg = 0
        else:
            num_diverg += 1
            
        if derr == 0:
            return True
        
        if num_diverg > max_iter:
            print('WARNING: GIVEN UP ON IMAGE SCALING!')
            return False
        
        