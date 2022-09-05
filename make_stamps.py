from astropy.io import fits,ascii; from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder; import numpy as np; import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch; import h5py; from pathlib import Path
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import CircularAperture,RectangularAperture

hdf5_dir=Path("hdf5/")
data_dir=Path("F:\\Trabajo de Fin de Grado 2021-2022\\Clasification\\fits")
# DAOStarFinder:
def Star_Finder(img, full_width=10.0, Threshold=75, Sigma=3, Iters=5, Plot=False, box_hsize=30):
    mean, median, std=sigma_clipped_stats(img, sigma=Sigma, maxiters=Iters)
    img_nobck=img-median; #peak_min=5000; peak_max=75000
    daofind=DAOStarFinder(fwhm=full_width, threshold=Threshold*std) #, exclude_border=True, peakmax=peak_max
    sources=daofind(img_nobck); num_sources=len(sources)
    img_nobck=np.where(img_nobck>=std, img_nobck, 0) ## avoid negative values
    return img_nobck, sources, num_sources

def masks(img, sources, box_hsize=30): # for borders & close sources
    nx=img.shape[1]; ny=img.shape[0]
    mask_valid=(sources['xcentroid']>(box_hsize+2))&(sources['xcentroid']<(nx-box_hsize-2))\
      &(sources['ycentroid']>(box_hsize+2))&(sources['ycentroid']<(ny-box_hsize-2))
    _mask_valid=[(i,j) for i,j in zip(sources['xcentroid'],sources['ycentroid'])]
    _mask_valid2=[]; mask_valid2=[True]*len(mask_valid)

    for i1, j1 in enumerate(_mask_valid):
      for i2, j2 in enumerate(_mask_valid):
        if i2!=i1 and np.abs(j2[0]-j1[0])<(2*box_hsize) and np.abs(j2[1]-j1[1])<(2*box_hsize):
          if (i1 in _mask_valid2) or (i2 in _mask_valid2):
            continue
          _mask_valid2.append(i1); _mask_valid2.append(i2)
    _mask_valid2=sorted(_mask_valid2)
    for i in _mask_valid2:
      mask_valid2[i]=False
    sources=sources[mask_valid&mask_valid2]
    return sources

def extract_sources_donuts(img, box_hsize=30, Sigma=3, Iters=5, Class='Donut', ImName='ImName'):
    mean, median, std=sigma_clipped_stats(img, sigma=Sigma, maxiters=Iters)
    img_nobck=img-median; #peak_min=5000; peak_max=75000
    img_nobck=np.where(img_nobck>=std, img_nobck, 0) ## avoid negative values
    sources=np.array([[781,305], [2358,420], [2677,431], [2814,789], [1728,1571],
        [1538,1938], [1656,926], [1211,1568]]); peak_max=75000; #peak_min=5000;
    xsources=np.zeros([sources.shape[0],2]); ysources=np.zeros([sources.shape[0],2])
    for i in range(sources.shape[0]):
        xsources[i]=sources[i][0]; ysources[i]=sources[i][1]

    pos=[(x,y) for x,y in zip(xsources, ysources)]; pos=np.array(pos).astype(np.int32)
    nstamps=sources.shape[0]; ## now retrieve the stamps
    img_stamps=np.zeros([nstamps,box_hsize*2,box_hsize*2]); img_meta=[]; img_class=Class
    contador=0; i=0;
    for j in range(nstamps):
        xc=pos[j][0][0]; yc=pos[j][1][0];
        k=img_nobck[yc-box_hsize:yc+box_hsize,xc-box_hsize:xc+box_hsize]
        if np.amax(k)>peak_max:
            i+=1; continue;
        img_stamps[j-i,:,:]=k
        contador+=1; img_stamps[j-i,:,:]=img_stamps[j-i,:,:]/np.amax(img_stamps[j-i,:,:])
        img_meta.append([img_class,xc,yc,2*box_hsize]) #[ImName,img_class,xc,yc,2*box_hsize]
    for o in range(i):
        img_stamps=np.delete(img_stamps, -1, axis=0)
    _img_meta=np.array(img_meta); print('Extracted #'+str(contador)+' stamps of '+ImName)
    return {'stamps':img_stamps, 'labels':_img_meta, 'nstamps':contador}

def extract_sources(img, full_width=10.0, Threshold=75,Sigma=3,Iters=5,Plot=False,Class='INDEF',box_hsize=30,ImName='ImName'):
    n_stamps=3; peak_min=10; peak_max=75000;
    img_nobck, _sources, nstamps=Star_Finder(img)
    
    if nstamps<=n_stamps:
        img_nobck, _sources, nstamps=Star_Finder(img, full_width=8.0, Threshold=50)
    sources=masks(img_nobck, _sources)
    
    pos=[(x,y) for x,y in zip(sources['xcentroid'],sources['ycentroid'])] 
    nstamps=len(sources); contador=0; i=0; ## now retrieve the stamps
    img_stamps=np.zeros([nstamps,box_hsize*2,box_hsize*2]); img_meta=[]; img_class=Class
    
    for j in range(nstamps):
        xc=int(np.round(pos[j][0])); yc=int(np.round(pos[j][1]))
        k=img_nobck[yc-box_hsize:yc+box_hsize,xc-box_hsize:xc+box_hsize]
        if np.amax(k)>peak_max or np.amax(k)<peak_min: #or np.amax(k)<peak_min
            i+=1; continue;
        img_stamps[j-i,:,:]=k
        contador+=1; img_stamps[j-i,:,:]=img_stamps[j-i,:,:]/np.amax(img_stamps[j-i,:,:])
        img_meta.append([img_class,xc,yc,2*box_hsize]) #[ImName,img_class,xc,yc,2*box_hsize]
    for o in range(i):
        img_stamps=np.delete(img_stamps, -1, axis=0)
    _img_meta=np.array(img_meta); print('Extracted #'+str(contador)+' stamps of '+ImName)
    mean, median, std=sigma_clipped_stats(img, sigma=Sigma, maxiters=Iters)
    if Plot:
        apertures=RectangularAperture(pos, 2*box_hsize, 2*box_hsize)
        norm=ImageNormalize(stretch=SqrtStretch(),vmin=median-2*std,vmax=median+15*std)
        plt.figure(); plt.imshow(img, cmap='Greys', origin='lower', norm=norm)
        apertures.plot(color='blue', lw=1.5, alpha=0.5); plt.show(block=True)
    return {'stamps':img_stamps, 'labels':_img_meta, 'nstamps':contador}

def read_many_hdf5_webt(fname,hdf5_dir='./'):
    # Open the HDF5 file
    file=h5py.File(hdf5_dir+fname+".h5", "r+")
    stamps=np.array(file["/stamps"]); labels=np.array(file["/labels"]).astype('S10')
    return stamps, labels
def store_many_hdf5_webt(fname,stamps,labels,hdf5_dir='./'):
    file=h5py.File(hdf5_dir+fname+".h5", "w")
    # Create a dataset in the file
    dataset=file.create_dataset("stamps", np.shape(stamps), h5py.h5t.IEEE_F32LE, data=stamps)
    labels_set=file.create_dataset("labels", np.shape(labels), h5py.h5t.IEEE_F32LE, data=labels); file.close() #'S10'

def multiply_images(all_stamps,all_labels,src_dict):
    #if f['class']==0: # rotate 'Circulares'
        #all_stamps=np.vstack((all_stamps,src_dict['stamps'][:,::-1,::-1]))
        #all_labels=np.vstack((all_labels,src_dict['labels']))
    if f['class']==1: # rotate 'Cometas'
        all_stamps=np.vstack((all_stamps,src_dict['stamps'][:,::-1,::-1]))
        all_stamps=np.vstack((all_stamps,src_dict['stamps'][:,::-1,:])) # add rotated
        all_stamps=np.vstack((all_stamps,src_dict['stamps'][:,:,::-1]))
        all_labels=np.vstack((all_labels,src_dict['labels']))
        all_labels=np.vstack((all_labels,src_dict['labels']))
        all_labels=np.vstack((all_labels,src_dict['labels']))
    elif f['class']==2: # rotate 'Donuts'
        all_stamps=np.vstack((all_stamps,src_dict['stamps'][:,::-1,:]))
        all_stamps=np.vstack((all_stamps,src_dict['stamps'][:,:,::-1]))
        all_stamps=np.vstack((all_stamps,src_dict['stamps'][:,::-1,::-1]))
        all_labels=np.vstack((all_labels,src_dict['labels']))
        all_labels=np.vstack((all_labels,src_dict['labels']))
        all_labels=np.vstack((all_labels,src_dict['labels']))

    return all_stamps, all_labels #{}
# 'Donut' images:
img1='tfn0m410-kb23-20190527-0441-e91.fits.fz'; img2='tfn0m410-kb23-20190527-0442-e91.fits.fz'; 
img3='tfn0m410-kb23-20190527-0443-e91.fits.fz'
## read list of files
imfits_list=ascii.read('imglist_training.lst') #imglist_validation, imglist_evaluation
## browse files
a=0; b=0; c=0
for f in imfits_list:
    hdu=fits.open(data_dir/f['Filename']); img=hdu[1].data
    if f['Filename']==img1 or f['Filename']==img2 or f['Filename']==img3:
        src_dict=extract_sources_donuts(img,ImName=f['Filename'],Class=f['class'])
        c+=src_dict['nstamps']
    else:
        src_dict=extract_sources(img,ImName=f['Filename'],Class=f['class'],Plot=False)
        if f['class']==0:
            a+=src_dict['nstamps']
        else:
            b+=src_dict['nstamps']
    try:
        all_stamps=np.vstack((all_stamps,src_dict['stamps']))
        all_labels=np.vstack((all_labels,src_dict['labels']))
        #all_stamps, all_labels=multiply_images(all_stamps,all_labels,src_dict) # rotate images
    except:
        all_stamps=src_dict['stamps']
        if src_dict['nstamps']==0: ## to unbroke -> all_labels
            continue;
        all_labels=src_dict['labels']
        #all_stamps, all_labels=multiply_images(all_stamps,all_labels,src_dict)

store_many_hdf5_webt('training', all_stamps, all_labels) #.h5
print('Extracted #'+str(all_stamps.shape[0])+' stamps')
print("Extracted #"+str(a)+
    " stamps of 'Circulares', #"+str(b)+" stamps of 'Cometas', #"+
    str(c)+" stamps of 'Donuts'") #4*b & 4*c if -> multiply_images
