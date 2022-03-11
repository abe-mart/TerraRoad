import streamlit as st
from PIL import Image
import numpy as np
import io
import imageio
from scipy import ndimage
from skimage.draw import polygon
from skimage.transform import resize
from svgpathtools import svg2paths2, Line, Path
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter
import os
import tempfile

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

# Streamlit image reading and writing (with caches for performance)
@st.cache
def load_image(image_file):
    array = imageio.imread(image_file)
    img = Image.fromarray(array).convert('RGB')
    return img

@st.cache
def load_array(image_file):
    st.write(image_file)
    array = imageio.imread(image_file)
    # img = Image.fromarray(array).convert('RGB')
    return array

# @st.cache
def load_svg(svg_file):
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp,'temp_file.svg'),"wb") as f:
            f.write(file_name_road.getbuffer())
        paths, attributes, svg_attributes = svg2paths2(os.path.join(tmp,'temp_file.svg'))
    return paths, attributes, svg_attributes

@st.cache
def prep_jpg_download(image):
    temp = io.BytesIO()
    image.save(temp, format="jpeg")
    return temp

@st.cache 
def prep_exr_download(image_file):
    array = imageio.imread(image_file)
    temp = io.BytesIO()
    imageio.imwrite(temp,array.astype(np.float32),format='EXR')
    return temp

@st.cache 
def prep_array_download(array):
    # array = imageio.imread(image_file)
    temp = io.BytesIO()
    imageio.imwrite(temp,array.astype(np.float32),format='TIF')
    return temp

def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://images.unsplash.com/photo-1536420100273-cabfa8e5b67a?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# %% Begin Page Layout
    
st.title('TerraRoad')

# File Upload
st.subheader('Input Files')
col1, col2 = st.columns(2)
with col1:
    file_name_ter = st.file_uploader('Terrain File (.exr)',type='exr')
with col2:
    file_name_road = st.file_uploader('Road Shape File (.svg)',type='svg')

# Basic Settings
with st.expander('Basic Settings', expanded=False):
    col1,col2,col3 = st.columns(3)
    with col1:
        road_width = st.slider('Road Width',5,100,20)
        border_width = st.slider('Shoulder Width',0,200,100)
        verge_width = st.slider('Verge Width',5,400,200)
    with col2:
        elevation_smoothing = st.slider('Elevation Smoothing',25,300,150)
        center_line_width = st.slider('Center Line Width',1,5,2)
    with col3:
        side_line_width = st.slider('Side Line Width',1,5,2)
        side_line_offset = st.slider('Side line offset',1,100,8)
        
# Advanced Settings
with st.expander('Advanced Settings', expanded=False):
    col1,col2,col3 = st.columns(3)
    with col1:
        road_segments = st.slider('Road Segments',5,500,400)
        edge_segments = st.slider('Edge Segments',5,100,25)
        elevation_smoothing_local = st.slider('Local Elevation Smoothing',1,5,3)
    with col2:
        shoulder_smoothing = st.slider('Shoulder Smoothing',1,10,5)
        dash_spacing = st.slider('Dash Spacing',1,5,3)
    with col3:
        dash_mult = st.slider('Dash Multiplier',1,50,20)
        road_altitude_offset = st.slider('Road Alititude Offset',-100,100,0)
        
# Mask settings at defaults for now
texture_upscale = 1
      
generate = st.button('Process Road')
if generate:
    
    # Read Terrain
    print('Importing Terrain')
    st.write('Importing Terrain')
    mat = load_array(file_name_ter)
    
    mat0 = np.copy(mat)
    st.write(mat.shape)
    
    # Read Road path vector
    print('Importing Road Path')
    st.write('Importing Road Path')
    paths, attributes, svg_attributes = load_svg(file_name_road)
    P = paths[0]
    
    # Sample pixels along path
    print('Extracting center line')
    st.write('Extracting center line')
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
    # rmask[road[:,1],road[:,0]] = 1
    
    # Extract path altitudes
    pathval = mat[road[:,1],road[:,0]]
    
    # Offset path altitude (rivers, embankments)
    pathval = pathval + road_altitude_offset
    
    # Smooth elevation along road path
    roadsmth = ndimage.gaussian_filter1d(pathval, elevation_smoothing)
    
    # Setup masks
    # mask_info = {}
    # for mask in mask_names:
    #     mask_info[mask] = {}
    #     mask_info[mask]['name'] = mask
    #     mask_info[mask]['upscale'] = values[mask+'scale']
    #     mask_info[mask]['blur'] = values[mask+'blur']
    #     mask_info[mask]['active'] = values[mask+'check']
    #     mask_info[mask]['format'] = values[mask+'format']
    #     if mask in ['road','center_line','center_dashes','side_lines','verge']:
    #         mask_info[mask]['mat'] = np.zeros(np.multiply(mat.shape,mask_info[mask]['upscale']))
    #     else:
    #         mask_info[mask]['mat'] = np.zeros(mat.shape)
    
    # Get road and shoulder surfaces
    fmask = np.zeros(mat.shape)
    rmask = np.zeros(mat.shape)
    rbmask = np.zeros(mat.shape)
    
    # Get road and shoulder edges
    rL = offset_curve(P,road_width/2,edge_segments)
    rR = offset_curve(P,-road_width/2,edge_segments)
    rLL = offset_curve(P,border_width/2,edge_segments)
    rRR = offset_curve(P,-border_width/2,edge_segments)
    vL = offset_curve(P,verge_width/2,edge_segments)
    vR = offset_curve(P,-verge_width/2,edge_segments)
    
    st.write('Processing Road')
    pbar = st.progress(0)
    # window['STATUS'].update('Building Road')
    for i in range(len(rL)):
        # progress = sg.OneLineProgressMeter('Road Progress', i+1, len(rL),  '', 'Building Road',orientation='h',key='build_progress')
        # if progress == False:
        #     break
        pbar.progress(i/len(rL))
        
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
        # mask_info[mask]['mat'][rsx,rsy] = 1
        rbmask[rsx,rsy] = 1
        
        # Verge
        mask = 'verge'
        bp1 = vL[i].point(np.linspace(0,1,3))
        bpx1 = np.real(bp1)
        bpy1 = np.imag(bp1)
        bp2 = vR[i].point(np.linspace(0,1,3))
        bpx2 = np.flip(np.real(bp2))
        bpy2 = np.flip(np.imag(bp2))
        # Higher res road surface
        # texture_upscale = mask_info[mask]['upscale']
        bpx1 = bpx1 * texture_upscale
        bpx2 = bpx2 * texture_upscale
        bpy1 = bpy1 * texture_upscale
        bpy2 = bpy2 * texture_upscale
        rsyBig,rsxBig = np.clip(polygon(np.concatenate([bpx1,bpx2]),np.concatenate([bpy1,bpy2])),0,len(mat)*texture_upscale-1)
        # mask_info[mask]['mat'][rsxBig,rsyBig] = 1
        
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
        # texture_upscale = mask_info[mask]['upscale']
        bpx1 = bpx1 * texture_upscale
        bpx2 = bpx2 * texture_upscale
        bpy1 = bpy1 * texture_upscale
        bpy2 = bpy2 * texture_upscale
        rsyBig,rsxBig = np.clip(polygon(np.concatenate([bpx1,bpx2]),np.concatenate([bpy1,bpy2])),0,len(mat)*texture_upscale-1)
        # mask_info[mask]['mat'][rsxBig,rsyBig] = 1
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
        # mask_info['shoulder_fade']['mat'][rss[:,0],rss[:,1]] = fade
        # ffmask[rss[:,0],rss[:,1]] = ffull
        # mask_info['road_and_shlder_fade']['mat'][rss[:,0],rss[:,1]] = ffull
        
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
        # mask_info['cut']['mat'][rss[cutidx,0],rss[cutidx,1]] = 1
        fillidx = mat[rss[:,0],rss[:,1]] < roadsmth[idx_min]
        # fillmask[rss[fillidx,0],rss[fillidx,1]] = 1
        # mask_info['fill']['mat'][rss[fillidx,0],rss[fillidx,1]] = 1
        
        # Apply road surface
        mat[rsx,rsy] = hvals
        
    # Smooth road surface and shoulder
    st.write('Smoothing Road Surface')
    # window['STATUS'].update('Smoothing Road Surface')
    mat_blur = gaussian_filter(mat,sigma=shoulder_smoothing)
    mat = np.where(rbmask>0,mat_blur,mat)
    mix_mask = np.maximum(fmask,rmask)
    mat_out = mat*mix_mask + mat0*(1-mix_mask)
    
    # Write Terrain
    st.write('Exporting Terrain')
    
    temp_ter = prep_array_download(mat_out)
    st.download_button('Download Finished Terrain', temp_ter.getvalue(),'terrain_with_road.tif')
    
# Mask Settings
# with st.expander('Mask Settings', expanded=False): 
#     col = st.columns(5)
#     with col[0]:
#         st.checkbox('Road')
#     with col[1]:
#         st.number_input()
# set_bg_hack_url()

# if image_file is not None:
#     im = load_image(image_file)
#     st.image(im)
    
#     temp = prep_jpg_download(im)
#     st.download_button('Download Image', temp.getvalue(),'im.jpg')
    
#     temp2 = prep_exr_download(image_file)
#     st.download_button('Download Image EXR', temp2.getvalue(),'im2.exr')
    
    