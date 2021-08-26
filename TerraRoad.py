# -*- coding: utf-8 -*-

# Import needed packages
import PySimpleGUI as sg
import imageio
import numpy as np
from scipy import ndimage
from skimage.draw import polygon
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
          [sg.Text('Terrain (EXR)', size=(15, 1)), sg.InputText(), sg.FileBrowse(key='file_name_ter',file_types=(("EXR Terrain Files", "*.exr"),),enable_events=True)],
          [sg.Text('Road Shape (SVG) ', size=(15, 1)), sg.InputText(), sg.FileBrowse(key='file_name_road',file_types=(("Road Path SVG", "*.svg"),))],
          [sg.Button('Save Terrain as JPEG',key='jpg'),sg.Text('Select Terrain and Output Folder first.',key='jpgtext')],
          [sg.Text('Select Output Folder',font='underline')],
          [sg.Text('Output Folder', size=(15, 1)), sg.InputText(), sg.FolderBrowse(key='path')],
          [sg.Text('Settings')]]

# Basic Settings
layout1 = []
settings = [['road_width',5,100,20],
            ['shoulder_width',0,200,100],
            ['elevation_smoothing',25,300,150],
            ['texture_upscale',1,5,3],
            ['dash_spacing',1,5,3]]
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
            ['center_line_width',1,5,1],
            ['side_line_width',1,5,2],
            ['side_line_offset',1,100,8],
            ['mask_smoothing',1,10,5],
            ['dash_multiplier',1,50,20]]
for row in settings:
    setting = row[0]
    minV = row[1]
    maxV = row[2]
    default = row[3]
    layout2 += [[sg.Text(setting.replace('_',' ').title(), size=(20,1)), sg.Slider(range=(minV,maxV),default_value=default,orientation='horizontal',key=setting)]]
    
tbgrp = [[sg.TabGroup([[sg.Tab('Basic Settings',layout1),sg.Tab('Advanced Settings',layout2)]])]]   

layout += [[tbgrp]]
    
layout += [[sg.Button('Create Road'), sg.Button('Exit')]]
layout += [[sg.Text('Awaiting Input',key='STATUS')]]

window = sg.Window('TerraRoad', layout)


# Run GUI
while True:  # Event Loop
    event, values = window.read()
    print(event, values)
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
        texture_upscale = int(values['texture_upscale'])
        center_line_width = int(values['center_line_width'])
        side_line_width = int(values['side_line_width'])
        side_line_offset = int(values['side_line_offset'])
        mask_smoothing = int(values['mask_smoothing'])
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
        rmask[road[:,1],road[:,0]] = 1
        
        
        # Extract path altitudes
        pathval = mat[road[:,1],road[:,0]]
        
        # Smooth elevation along road path
        roadsmth = ndimage.gaussian_filter1d(pathval, elevation_smoothing)
        
        # Get road and shoulder surfaces
        bmask = np.zeros(rmask.shape)
        fmask = np.zeros(rmask.shape)
        ffmask = np.zeros(rmask.shape)
        rbmask = np.zeros(rmask.shape)
        cutmask = np.zeros(rmask.shape)
        fillmask = np.zeros(rmask.shape)
        rmaskBig = np.zeros(np.multiply(rmask.shape,texture_upscale))
        rcmaskBig = np.zeros(np.multiply(rmask.shape,texture_upscale))
        rcmaskBigDash = np.zeros(np.multiply(rmask.shape,texture_upscale))
        rsmaskBig = np.zeros(np.multiply(rmask.shape,texture_upscale))
        
        # Get road and shoulder edges
        rL = offset_curve(P,road_width/2,edge_segments)
        rR = offset_curve(P,-road_width/2,edge_segments)
        rLL = offset_curve(P,border_width/2,edge_segments)
        rRR = offset_curve(P,-border_width/2,edge_segments)
        
        print('Processing Road')
        window['STATUS'].update('Building Road')
        for i in range(len(rL)):
            progress = sg.OneLineProgressMeter('My Meter', i+1, len(rL),  '', 'Building Road',orientation='h',key='build_progress')
            if progress == False:
                break
            
            # Shoulder
            bp1 = rLL[i].point(np.linspace(0,1,3))
            bpx1 = np.real(bp1)
            bpy1 = np.imag(bp1)
            bp2 = rRR[i].point(np.linspace(0,1,3))
            bpx2 = np.flip(np.real(bp2))
            bpy2 = np.flip(np.imag(bp2))
            rsy,rsx = np.clip(polygon(np.concatenate([bpx1,bpx2]),np.concatenate([bpy1,bpy2])),0,len(mat)-1)
            rss = np.c_[rsx,rsy]
            bmask[rsx,rsy] = 1
            rbmask[rsx,rsy] = 1
            
            # Road Surface
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
            bpx1 = bpx1 * texture_upscale
            bpx2 = bpx2 * texture_upscale
            bpy1 = bpy1 * texture_upscale
            bpy2 = bpy2 * texture_upscale
            rsyBig,rsxBig = np.clip(polygon(np.concatenate([bpx1,bpx2]),np.concatenate([bpy1,bpy2])),0,len(mat)*texture_upscale-1)
            rmaskBig[rsxBig,rsyBig] = 1
            
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
            ffmask[rss[:,0],rss[:,1]] = ffull
            
            # Get road surface distances to center line (TODO: REDUNDANT, TAKE DISTANCES FROM ABOVE)
            dist_r = distance.cdist(np.c_[rs[:,1],rs[:,0]],road,'euclidean')
            
            # Heights for road surface
            cloInd = np.argsort(dist_r,axis=1)
            hvals = np.mean(roadsmth[cloInd[:,0:elevation_smoothing_local]],axis=1)
            
            # Apply shoulder
            mat[rss[:,0],rss[:,1]] = roadsmth[idx_min]*fade + mat[rss[:,0],rss[:,1]]*(1-fade)
            
            # Filter for cut and fill masks
            cutidx = mat[rss[:,0],rss[:,1]] > roadsmth[idx_min]
            cutmask[rss[cutidx,0],rss[cutidx,1]] = 1
            fillidx = mat[rss[:,0],rss[:,1]] < roadsmth[idx_min]
            fillmask[rss[fillidx,0],rss[fillidx,1]] = 1
            
            # Apply road surface
            mat[rsx,rsy] = hvals
           
        # Make sure progress bar window closed
        sg.OneLineProgressMeterCancel('build_progress')
        window['STATUS'].update('Starting textures')
        
        # Dashed Center Line
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
            bp1 = rcL[i].point(np.linspace(0,1,3))
            bpx1 = np.real(bp1)
            bpy1 = np.imag(bp1)
            bp2 = rcR[i].point(np.linspace(0,1,3))
            bpx2 = np.flip(np.real(bp2))
            bpy2 = np.flip(np.imag(bp2))
            # Higher res center line
            bpx1 = bpx1 * texture_upscale
            bpx2 = bpx2 * texture_upscale
            bpy1 = bpy1 * texture_upscale
            bpy2 = bpy2 * texture_upscale
            rsyBig,rsxBig = np.clip(polygon(np.concatenate([bpx1,bpx2]),np.concatenate([bpy1,bpy2])),0,len(mat)*texture_upscale-1)
            rcmaskBigDash[rsxBig,rsyBig] = 1*dash[i]
            rcmaskBig[rsxBig,rsyBig] = 1
            
            # Side Line Left
            bp1 = rsLL[i].point(np.linspace(0,1,3))
            bpx1 = np.real(bp1)
            bpy1 = np.imag(bp1)
            bp2 = rsLR[i].point(np.linspace(0,1,3))
            bpx2 = np.flip(np.real(bp2))
            bpy2 = np.flip(np.imag(bp2))
            # Higher res line
            bpx1 = bpx1 * texture_upscale
            bpx2 = bpx2 * texture_upscale
            bpy1 = bpy1 * texture_upscale
            bpy2 = bpy2 * texture_upscale
            rsyBig,rsxBig = np.clip(polygon(np.concatenate([bpx1,bpx2]),np.concatenate([bpy1,bpy2])),0,len(mat)*texture_upscale-1)
            rsmaskBig[rsxBig,rsyBig] = 1
            
            # Side Line Right
            bp1 = rsRL[i].point(np.linspace(0,1,3))
            bpx1 = np.real(bp1)
            bpy1 = np.imag(bp1)
            bp2 = rsRR[i].point(np.linspace(0,1,3))
            bpx2 = np.flip(np.real(bp2))
            bpy2 = np.flip(np.imag(bp2))
            # Higher res line
            bpx1 = bpx1 * texture_upscale
            bpx2 = bpx2 * texture_upscale
            bpy1 = bpy1 * texture_upscale
            bpy2 = bpy2 * texture_upscale
            rsyBig,rsxBig = np.clip(polygon(np.concatenate([bpx1,bpx2]),np.concatenate([bpy1,bpy2])),0,len(mat)*texture_upscale-1)
            rsmaskBig[rsxBig,rsyBig] = 1
            
                    
        # Smooth road surface and shoulder
        print('Smoothing Road Surface')
        window['STATUS'].update('Smoothing Road Surface')
        mat_blur = gaussian_filter(mat,sigma=shoulder_smoothing)
        mat = np.where(rbmask>0,mat_blur,mat)
        mix_mask = np.maximum(fmask,rmask)
        mat_out = mat*mix_mask + mat0*(1-mix_mask)
        
        # Smooth masks
        print('Smoothing Masks')
        window['STATUS'].update('Smoothing Masks')
        rcmaskBig = gaussian_filter(rcmaskBig,mask_smoothing)
        rcmaskBigDash = gaussian_filter(rcmaskBigDash,mask_smoothing)
        rsmaskBig = gaussian_filter(rsmaskBig,mask_smoothing)
        rmaskBig = gaussian_filter(rmaskBig,mask_smoothing)
        
        # Write Terrain
        print('Exporting Terrain')
        window['STATUS'].update('Exporting Terrain')
        imageio.imwrite(join(path,'terrain_with_road.tif'),mat_out.astype(np.float32))
        
        # Write Masks
        print('Exporting Masks')
        window['STATUS'].update('Exporting Masks')
        imageio.imwrite(join(path,'masks_center_line.png'),(rcmaskBig*255).astype(np.uint8))
        imageio.imwrite(join(path,'masks_center_line_with_dashes.png'),(rcmaskBigDash*255).astype(np.uint8))
        imageio.imwrite(join(path,'masks_side_lines.png'),(rsmaskBig*255).astype(np.uint8))
        imageio.imwrite(join(path,'masks_road.png'),(rmaskBig*255).astype(np.uint8))
        imageio.imwrite(join(path,'masks_shoulder.png'),(bmask*255).astype(np.uint8))
        imageio.imwrite(join(path,'masks_shoulder_fade.png'),(fmask*255).astype(np.uint8))
        imageio.imwrite(join(path,'masks_road_and_shoulder_fade.png'),(ffmask*255).astype(np.uint8))
        imageio.imwrite(join(path,'masks_cut.png'),(cutmask*255).astype(np.uint8))
        imageio.imwrite(join(path,'masks_fill.png'),(fillmask*255).astype(np.uint8))
        print('Done')
        window['STATUS'].update('Done')
        
        # Open output folder
        webbrowser.open('file:///'+path)

window.close()