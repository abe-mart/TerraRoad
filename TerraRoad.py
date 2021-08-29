# -*- coding: utf-8 -*-

# Import needed packages
import PySimpleGUI as sg
import imageio
import numpy as np
from scipy import ndimage
from skimage.draw import polygon
# from skimage.filters import gaussian
from skimage.transform import resize
from svgpathtools import svg2paths2, Line, Path
from scipy.spatial import distance
# from sklearn.preprocessing import minmax_scale
from scipy.ndimage import gaussian_filter
from os.path import join, basename
import webbrowser

# Define utility functions for later
def smoothclamp(x, mi, mx): return mi + (mx-mi)*(lambda t: np.where(t < 0 , 0, np.where( t <= 1 , 3*t**2-2*t**3, 1 ) ) )( (x-mi)/(mx-mi) )

def minmax_scale(array):
    return (array - array.min(axis=0))/(array.max(axis=0) - array.min(axis=0))
    

def offset_curve(path, offset_distance, steps=1000):
    """Takes in a Path object, `path`, and a distance,
    `offset_distance`, and outputs an piecewise-linear approximation 
    of the 'parallel' offset curve."""
    nls = []
    for seg in path:
        # ct = 1
        for k in range(steps):
            t = k / steps
            offset_vector = offset_distance * seg.normal(t)
            nl = Line(seg.point(t), seg.point(t) + offset_vector)
            nls.append(nl)
    connect_the_dots = [Line(nls[k].end, nls[k+1].end) for k in range(len(nls)-1)]
    if path.isclosed():
        connect_the_dots.append(Line(nls[-1].end, nls[0].end))
    offset_path = Path(*connect_the_dots)
    return offset_path

def place(elem):
    '''
    Places element provided into a Column element so that its placement in the layout is retained.
    :param elem: the element to put into the layout
    :return: A column element containing the provided element
    '''
    return sg.Column([[elem]], pad=(0,0))

# Set up GUI layout
sg.theme('Dark Grey 2')  

layout = [[sg.Text('Select Input Files',font='underline')],
          [sg.Text('Terrain (EXR)', size=(15, 1)), sg.InputText(), sg.FileBrowse(key='file_name_ter',file_types=(("EXR Terrain Files", "*.exr"),),enable_events=True,change_submits=True)],
          [sg.Text('Road Shape (SVG) ', size=(15, 1)), sg.InputText(), sg.FileBrowse(key='file_name_road',file_types=(("Road Path SVG", "*.svg"),),change_submits=True)],
          [sg.Button('Save Terrain as JPEG',key='jpg'),sg.Text('Select Terrain and Output Folder first.',key='jpgtext')],
          [sg.Text('Select Output Folder',font='underline')],
          [sg.Text('Output Folder', size=(15, 1)), sg.InputText(), sg.FolderBrowse(key='path',change_submits=True)],
          [sg.Text('Settings')]]

# Basic Settings
layout1 = []
settings = [['road_width',5,100,20],
            ['shoulder_width',0,200,100],
            ['elevation_smoothing',25,300,150],
            ['center_line_width',1,5,2],
            ['side_line_width',1,5,2],
            ['side_line_offset',1,100,8]]
for row in settings:
    setting = row[0]
    minV = row[1]
    maxV = row[2]
    default = row[3]
    layout1 += [[sg.Text(setting.replace('_',' ').title(), size=(15,1)), sg.Slider(range=(minV,maxV),default_value=default,orientation='horizontal',key=setting)]]
    
# Advanced Settings
layout2 = []
settings = [['road_segments',5,500,400],
            ['edge_segments',5,100,25],
            ['local_elevation_smoothing',1,5,3],
            ['shoulder_smoothing',1,10,5],
            ['dash_spacing',1,5,3],
            ['dash_multiplier',1,50,20]]
for row in settings:
    setting = row[0]
    minV = row[1]
    maxV = row[2]
    default = row[3]
    layout2 += [[sg.Text(setting.replace('_',' ').title(), size=(20,1)), sg.Slider(range=(minV,maxV),default_value=default,orientation='horizontal',key=setting)]]
    
# Mask Settings
layout3 = []
mask_names = ['road','shoulder','shoulder_fade','road_and_shlder_fade','center_line','center_dashes','side_lines','cut','fill']
for mask in mask_names:
    smoothrange = [1,2,3,4,5,6,7,8,9,10]
    resrange = [1,2,3,4,5]
    layout3 += [[sg.Checkbox(mask.replace('_',' ').replace('and','&').title(),default=True,size=(15,1),key=mask+'check')
                 ,sg.Text('Blur'),sg.Spin(smoothrange,key=mask+'blur',initial_value=2),
                 sg.Text('Upscale'),sg.Spin(resrange,key=mask+'scale',change_submits=True),
                 sg.Text('Format'),sg.Combo(['PNG8','PNG16'],default_value='PNG16',key=mask+'format'),
                 sg.Text('Resolution: ',key=mask+'res')]]
    
tbgrp = [[sg.TabGroup([[sg.Tab('Basic Settings',layout1,key='tab1'),sg.Tab('Advanced Settings',layout2,key='tab2'),sg.Tab('Mask Settings',layout3,key='tab3')]],key='tabs',change_submits=True)]]   

layout += [[tbgrp]]
    
layout += [[sg.Button('Create Road'), sg.Button('Exit')]]
layout += [[sg.Text('Awaiting Input',key='STATUS')]]

window = sg.Window('TerraRoad', layout)

# EXR Read test completed?
exr_test = False

# Initialize matsize
matsize = False

# Run GUI
while True:  # Event Loop
    event, values = window.read()
    print(event, values)
    if exr_test == False:
        # Test to see if we can open an exr file
        try:
            exr = imageio.imread('./test_exr_read.exr')
        except:
            print('Could not read EXR image, attempting to install FreeImage plugin.')
            user_response = sg.popup_yes_no('Could not read test EXR image.  TerraRoad requires the FreeImage library to read EXR images.  Attempt to download FreeImage?  (See https://imageio.readthedocs.io/en/stable/format_exr-fi.html for more information).')
            if user_response == 'Yes':
                import imageio
                imageio.plugins.freeimage.download()
            else:
                print('Could not continue.  Please consult the documentation for more information.')
        exr_test = True
    if event == 'jpg':
        if not (values['file_name_ter'] and values['path']):
            print('ERROR')
            print('Please select terrain file and output folder first.')
        else:
            print(values['path'])
            # Read Terrain
            file_name_ter = values['file_name_ter']
            path = values['path']
            mat = imageio.imread(file_name_ter)
            
            # Normalize Terrain
            mat_norm = (mat + np.abs(np.min(mat)))/ np.max(mat + np.abs(np.min(mat)))
            mat_gray = mat_norm*255
            
            imname = basename(file_name_ter).split('.')[0] + '.jpg'
            imageio.imwrite(join(path,imname),(mat_gray).astype(np.uint8))
            window['jpgtext'].update('Saved as ' + imname)
    if event == 'file_name_ter' or event == 'path':
        if (values['file_name_ter'] and values['path']):
            window['jpgtext'] = 'Ready'
    if event == 'tabs' or event in [mask + 'scale' for mask in mask_names]:
        if values['file_name_ter'] and values['tabs'] == 'tab3':
            file_name_ter = values['file_name_ter']
            if matsize == False:
                mat = imageio.imread(file_name_ter)
                matsize = mat.shape
            for mask in mask_names:
                scale = values[mask+'scale']
                ter_scale = matsize[0]
                window[mask+'res'].update('Resolution: ' + str(ter_scale*scale) + ' x ' +str(ter_scale*scale))
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event == 'Create Road':  ######################## MAIN ROAD CREATOR SCRIPT
        window['STATUS'].update('Starting')
        
        # Get values from input form
        path = values['path']
        file_name_ter = values['file_name_ter']
        file_name_road = values['file_name_road']
        road_width = int(values['road_width'])
        border_width = int(values['shoulder_width'])
        road_segments = int(values['road_segments'])
        edge_segments = int(values['edge_segments'])
        elevation_smoothing = int(values['elevation_smoothing'])
        elevation_smoothing_local = int(values['local_elevation_smoothing'])
        shoulder_smoothing = int(values['shoulder_smoothing'])
        center_line_width = int(values['center_line_width'])
        side_line_width = int(values['side_line_width'])
        side_line_offset = int(values['side_line_offset'])
        dash_spacing = int(values['dash_spacing'])
        dash_mult = int(values['dash_multiplier'])
        
        # Read Terrain
        print('Importing Terrain')
        window['STATUS'].update('Importing Terrain')
        mat = imageio.imread(file_name_ter)
        mat0 = imageio.imread(file_name_ter)
        print(mat.shape)
        
        # Read Road path vector
        print('Importing Road Path')
        window['STATUS'].update('Importing Road Path')
        paths, attributes, svg_attributes = svg2paths2(file_name_road)
        P = paths[0]
        
        # Sample pixels along path
        print('Extracting center line')
        window['STATUS'].update('Extracting center line')
        rx = []
        ry = []
        for b in P:
            bp = b.point(np.linspace(0,1,road_segments))
            bpx = np.real(bp)
            bpy = np.imag(bp)
            rx.append(bpx)
            ry.append(bpy)
        
        rx = np.concatenate(rx)
        ry = np.concatenate(ry)
        road = np.clip((np.c_[rx,ry]).astype(int),0,len(mat)-1)
        
        rmask = np.zeros(mat.shape)
        
        # Extract path altitudes
        pathval = mat[road[:,1],road[:,0]]
        
        # Smooth elevation along road path
        roadsmth = ndimage.gaussian_filter1d(pathval, elevation_smoothing)
        
        # Setup masks
        mask_info = {}
        for mask in mask_names:
            mask_info[mask] = {}
            mask_info[mask]['name'] = mask
            mask_info[mask]['upscale'] = values[mask+'scale']
            mask_info[mask]['blur'] = values[mask+'blur']
            mask_info[mask]['active'] = values[mask+'check']
            mask_info[mask]['format'] = values[mask+'format']
            if mask in ['road','center_line','center_dashes','side_lines']:
                mask_info[mask]['mat'] = np.zeros(np.multiply(mat.shape,mask_info[mask]['upscale']))
            else:
                mask_info[mask]['mat'] = np.zeros(mat.shape)
        
        # Get road and shoulder surfaces
        # bmask = np.zeros(rmask.shape)
        fmask = np.zeros(mat.shape)
        # ffmask = np.zeros(rmask.shape)
        rmask = np.zeros(mat.shape)
        rbmask = np.zeros(mat.shape)
        # cutmask = np.zeros(rmask.shape)
        # fillmask = np.zeros(rmask.shape)
        # rmaskBig = np.zeros(np.multiply(rmask.shape,texture_upscale))
        # rcmaskBig = np.zeros(np.multiply(rmask.shape,texture_upscale))
        # rcmaskBigDash = np.zeros(np.multiply(rmask.shape,texture_upscale))
        # rsmaskBig = np.zeros(np.multiply(rmask.shape,texture_upscale))
        
        # Get road and shoulder edges
        rL = offset_curve(P,road_width/2,edge_segments)
        rR = offset_curve(P,-road_width/2,edge_segments)
        rLL = offset_curve(P,border_width/2,edge_segments)
        rRR = offset_curve(P,-border_width/2,edge_segments)
        
        print('Processing Road')
        window['STATUS'].update('Building Road')
        for i in range(len(rL)):
            progress = sg.OneLineProgressMeter('Road Progress', i+1, len(rL),  '', 'Building Road',orientation='h',key='build_progress')
            if progress == False:
                break
            
            # Shoulder
            mask = 'shoulder'
            bp1 = rLL[i].point(np.linspace(0,1,3))
            bpx1 = np.real(bp1)
            bpy1 = np.imag(bp1)
            bp2 = rRR[i].point(np.linspace(0,1,3))
            bpx2 = np.flip(np.real(bp2))
            bpy2 = np.flip(np.imag(bp2))
            rsy,rsx = np.clip(polygon(np.concatenate([bpx1,bpx2]),np.concatenate([bpy1,bpy2])),0,len(mat)-1)
            rss = np.c_[rsx,rsy]
            mask_info[mask]['mat'][rsx,rsy] = 1
            # bmask[rsx,rsy] = 1
            rbmask[rsx,rsy] = 1
            
            # Road Surface
            mask = 'road'
            bp1 = rL[i].point(np.linspace(0,1,3))
            bpx1 = np.real(bp1)
            bpy1 = np.imag(bp1)
            bp2 = rR[i].point(np.linspace(0,1,3))
            bpx2 = np.flip(np.real(bp2))
            bpy2 = np.flip(np.imag(bp2))
            rsy,rsx = np.clip(polygon(np.concatenate([bpx1,bpx2]),np.concatenate([bpy1,bpy2])),0,len(mat)-1)
            rs = np.c_[rsx,rsy]
            rmask[rsx,rsy] = 1
            rbmask[rsx,rsy] = 1
            # Higher res road surface
            texture_upscale = mask_info[mask]['upscale']
            bpx1 = bpx1 * texture_upscale
            bpx2 = bpx2 * texture_upscale
            bpy1 = bpy1 * texture_upscale
            bpy2 = bpy2 * texture_upscale
            rsyBig,rsxBig = np.clip(polygon(np.concatenate([bpx1,bpx2]),np.concatenate([bpy1,bpy2])),0,len(mat)*texture_upscale-1)
            mask_info[mask]['mat'][rsx,rsy] = 1
            # rmaskBig[rsxBig,rsyBig] = 1
            
            # Get shoulder distances to center line
            dist_s = distance.cdist(np.c_[rss[:,1],rss[:,0]],road,'euclidean')
            idx_min = np.argmin(dist_s,axis=1)
            dist_min = np.min(dist_s,axis=1)
            clamped = smoothclamp(dist_min,road_width/2,max(dist_min))
            clamped_scaled = minmax_scale(clamped)
            clipped = dist_min
            clipped[clipped < road_width/2] = road_width/2
            ffull = 1-minmax_scale(dist_min)
            fade = 1-clamped_scaled
            fmask[rss[:,0],rss[:,1]] = fade
            mask_info['shoulder_fade']['mat'][rss[:,0],rss[:,1]] = fade
            # ffmask[rss[:,0],rss[:,1]] = ffull
            mask_info['road_and_shlder_fade']['mat'][rss[:,0],rss[:,1]] = ffull
            
            # Get road surface distances to center line (TODO: REDUNDANT, TAKE DISTANCES FROM ABOVE)
            dist_r = distance.cdist(np.c_[rs[:,1],rs[:,0]],road,'euclidean')
            
            # Heights for road surface
            cloInd = np.argsort(dist_r,axis=1) # closest points on smoothed center line
            hvals = np.mean(roadsmth[cloInd[:,0:elevation_smoothing_local]],axis=1)
            
            # Apply shoulder
            mat[rss[:,0],rss[:,1]] = roadsmth[idx_min]*fade + mat[rss[:,0],rss[:,1]]*(1-fade)
            
            # Filter for cut and fill masks
            cutidx = mat[rss[:,0],rss[:,1]] > roadsmth[idx_min]
            # cutmask[rss[cutidx,0],rss[cutidx,1]] = 1
            mask_info['cut']['mat'][rss[cutidx,0],rss[cutidx,1]] = 1
            fillidx = mat[rss[:,0],rss[:,1]] < roadsmth[idx_min]
            # fillmask[rss[fillidx,0],rss[fillidx,1]] = 1
            mask_info['fill']['mat'][rss[fillidx,0],rss[fillidx,1]] = 1
            
            # Apply road surface
            mat[rsx,rsy] = hvals
           
        # Make sure progress bar window closed
        sg.OneLineProgressMeterCancel('build_progress')
        window['STATUS'].update('Starting textures')
        
        # Road Markings
        print('Dashing center line')
        window['STATUS'].update('Dashing center line')
        rcL = offset_curve(P,center_line_width/2,edge_segments*dash_mult)
        rcR = offset_curve(P,-center_line_width/2,edge_segments*dash_mult)   
        dash = np.zeros(len(rcL))
        dash[0:-1:dash_spacing] = 1
        rsLL = offset_curve(P,side_line_offset+side_line_width/2,edge_segments*dash_mult)
        rsLR = offset_curve(P,side_line_offset-side_line_width/2,edge_segments*dash_mult)
        rsRL = offset_curve(P,-side_line_offset+side_line_width/2,edge_segments*dash_mult)
        rsRR = offset_curve(P,-side_line_offset-side_line_width/2,edge_segments*dash_mult)
        for i in range(len(rcL)):
            # Center Line
            mask = 'center_line'
            bp1 = rcL[i].point(np.linspace(0,1,3))
            bpx1 = np.real(bp1)
            bpy1 = np.imag(bp1)
            bp2 = rcR[i].point(np.linspace(0,1,3))
            bpx2 = np.flip(np.real(bp2))
            bpy2 = np.flip(np.imag(bp2))
            # Higher res center line
            texture_upscale = mask_info[mask]['upscale']
            bpx1 = bpx1 * texture_upscale
            bpx2 = bpx2 * texture_upscale
            bpy1 = bpy1 * texture_upscale
            bpy2 = bpy2 * texture_upscale
            rsyBig,rsxBig = np.clip(polygon(np.concatenate([bpx1,bpx2]),np.concatenate([bpy1,bpy2])),0,len(mat)*texture_upscale-1)
            # rcmaskBig[rsxBig,rsyBig] = 1
            mask_info['center_line']['mat'][rsxBig,rsyBig] = 1
            
            # Center Line with dashes
            mask = 'center_dashes'
            bp1 = rcL[i].point(np.linspace(0,1,3))
            bpx1 = np.real(bp1)
            bpy1 = np.imag(bp1)
            bp2 = rcR[i].point(np.linspace(0,1,3))
            bpx2 = np.flip(np.real(bp2))
            bpy2 = np.flip(np.imag(bp2))
            # Higher res center line
            texture_upscale = mask_info[mask]['upscale']
            bpx1 = bpx1 * texture_upscale
            bpx2 = bpx2 * texture_upscale
            bpy1 = bpy1 * texture_upscale
            bpy2 = bpy2 * texture_upscale
            rsyBig,rsxBig = np.clip(polygon(np.concatenate([bpx1,bpx2]),np.concatenate([bpy1,bpy2])),0,len(mat)*texture_upscale-1)
            # rcmaskBigDash[rsxBig,rsyBig] = 1*dash[i]
            mask_info['center_dashes']['mat'][rsxBig,rsyBig] = 1*dash[i]
            
            # Side Line Left
            mask = 'side_lines'
            bp1 = rsLL[i].point(np.linspace(0,1,3))
            bpx1 = np.real(bp1)
            bpy1 = np.imag(bp1)
            bp2 = rsLR[i].point(np.linspace(0,1,3))
            bpx2 = np.flip(np.real(bp2))
            bpy2 = np.flip(np.imag(bp2))
            # Higher res line
            texture_upscale = mask_info[mask]['upscale']
            bpx1 = bpx1 * texture_upscale
            bpx2 = bpx2 * texture_upscale
            bpy1 = bpy1 * texture_upscale
            bpy2 = bpy2 * texture_upscale
            rsyBig,rsxBig = np.clip(polygon(np.concatenate([bpx1,bpx2]),np.concatenate([bpy1,bpy2])),0,len(mat)*texture_upscale-1)
            # rsmaskBig[rsxBig,rsyBig] = 1
            mask_info['side_lines']['mat'][rsxBig,rsyBig] = 1
            
            # Side Line Right
            mask = 'side_lines'
            bp1 = rsRL[i].point(np.linspace(0,1,3))
            bpx1 = np.real(bp1)
            bpy1 = np.imag(bp1)
            bp2 = rsRR[i].point(np.linspace(0,1,3))
            bpx2 = np.flip(np.real(bp2))
            bpy2 = np.flip(np.imag(bp2))
            # Higher res 
            texture_upscale = mask_info[mask]['upscale']
            bpx1 = bpx1 * texture_upscale
            bpx2 = bpx2 * texture_upscale
            bpy1 = bpy1 * texture_upscale
            bpy2 = bpy2 * texture_upscale
            rsyBig,rsxBig = np.clip(polygon(np.concatenate([bpx1,bpx2]),np.concatenate([bpy1,bpy2])),0,len(mat)*texture_upscale-1)
            # rsmaskBig[rsxBig,rsyBig] = 1
            mask_info['side_lines']['mat'][rsxBig,rsyBig] = 1
            
                    
        # Smooth road surface and shoulder
        print('Smooting Road Surface')
        window['STATUS'].update('Smoothing Road Surface')
        mat_blur = gaussian_filter(mat,sigma=shoulder_smoothing)
        mat = np.where(rbmask>0,mat_blur,mat)
        mix_mask = np.maximum(fmask,rmask)
        mat_out = mat*mix_mask + mat0*(1-mix_mask)
        
        # Write Terrain
        print('Exporting Terrain')
        window['STATUS'].update('Exporting Terrain')
        imageio.imwrite(join(path,'terrain_with_road.tif'),mat_out.astype(np.float32))
        
        # Upscale remaining masks
        for mask_name in mask_info:
            if mask_name not in ['road','center_line','center_dashes','side_lines']:
                mask = mask_info[mask_name]
                mask['mat'] = resize(mask['mat'],np.multiply(mat.shape,mask['upscale']))
        
        # Blur masks
        print('Blurring Masks')
        window['STATUS'].update('Blurring Masks')
        for mask_name in mask_info:
            mask = mask_info[mask_name]
            if mask['active']:
                print('Blurring ' + mask_name)
                mask['mat'] = gaussian_filter(mask['mat'],mask['blur'])

        # Write Masks
        print('Exporting Masks')
        window['STATUS'].update('Exporting Masks')
        for mask_name in mask_info:
            mask = mask_info[mask_name]
            print('Saving Mask: ' + mask['name'])
            if mask['active']:
                if mask['format'] == 'PNG8':
                    fmt = '.png'
                    typ = np.uint8
                    mult = 255
                elif mask['format'] == 'PNG16':
                    fmt = '.png'
                    typ = np.uint16
                    mult = 65535
                filename = 'masks_' + mask['name'] + fmt
                # array = mask_arrays[mask]
                imageio.imwrite(join(path,filename),(mask['mat']*mult).astype(typ))
                
        print('Done')
        window['STATUS'].update('Done')
        
        # Open output folder
        webbrowser.open('file:///'+path)

window.close()
