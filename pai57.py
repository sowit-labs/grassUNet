import numpy as np
from PIL import Image, ImageDraw 
import math

import argparse
import os
import sys
import glob
from tqdm import tqdm

from string import Template

from datetime import date

class DP57:
    def __init__(self,mask_path,vegColor=255,skyColor=0,gapColor=100):
        """Initialize the class with the parameters of the mask"""
        
        self.mask_path=mask_path
        self.vegColor=vegColor
        self.skyColor=skyColor
        self.gapColor=gapColor
        
        self.mask = Image.open(self.mask_path).convert('L')
        w,h = self.mask.size
        
        self.param(focal_length=0,
                        img_sensor_w=0,
                        img_sensor_h=0,
                        fov=10,#the field of view to take account
                        cell_size=30)
    
    def param(self,focal_length,img_sensor_w,img_sensor_h,fov,cell_size=30):
        """Configure new parameter values"""
        self.focal_length=focal_length
        self.img_sensor_w=img_sensor_w
        self.img_sensor_h=img_sensor_h
        self.fov=fov
        self.cell_size=cell_size
        
        self.preprocess()
    
    def preprocess(self):
        """Mask the up and down borders according to optical correction"""
        if self.focal_length>0 and self.img_sensor_w>0 and self.img_sensor_h>0:
            vfov=2*math.degrees(math.atan(self.img_sensor_h/(2*self.focal_length)))
            gap_band_height=(self.mask.size[1]-self.mask.size[1]*2*abs(self.fov)/vfov)/2

            mask = Image.open(self.mask_path).convert('L')
            bandDrawing = ImageDraw.Draw(mask)
            bandDrawing.rectangle([0,0,mask.size[0],max(0,gap_band_height)],fill=self.gapColor)
            bandDrawing.rectangle([0,min(mask.size[1]-gap_band_height,mask.size[1]),mask.size[0],mask.size[1]],fill=self.gapColor)
            self.mask = mask
        
    def P0(self):
        """Compute the gap fraction of the whole image"""
        mask = np.array(self.mask, dtype=np.dtype('uint8'))
        vegNb = np.count_nonzero(mask==self.vegColor)
        skyNb = np.count_nonzero(mask==self.skyColor)
        return skyNb/(vegNb+skyNb) if (vegNb+skyNb)!=0 else 0
    
    def PAI57(self):
        """Compute the effective PAI value of the whole image according to Warren-Wilson formulation (1963)"""
        return -2*math.log(self.P0())*math.cos(math.radians(57.5))
    
    def PAItrue(self, PAIsat=10):
        """Compute the true PAI value taking account the clumping effect according to Lang-Xiang formulation (1986)"""
        w, h = self.mask.size
        hcell = self.cell_size
        wcell = self.cell_size
        
        mask=np.array(self.mask,dtype=np.uint8)
        cells=[mask[i*hcell:min((i+1)*hcell,h),j*wcell:min((j+1)*wcell,w)] for i in range(math.ceil(h/hcell)) for j in range(math.ceil(w/wcell))]
                
        P0Cells = [np.count_nonzero(cell==self.skyColor)/(np.count_nonzero(cell==self.vegColor)+np.count_nonzero(cell==self.skyColor)) for cell in cells]
        P0sat=math.exp(-0.5*PAIsat/math.cos(math.radians(57.5)))
        P0Cells = [P0Cell if P0Cell!=0 else P0sat for P0Cell in P0Cells]
                
        CF_LX=math.log(np.mean(P0Cells))/np.mean([math.log(P0Cell) for P0Cell in P0Cells])
                
        return self.PAI57()/CF_LX


def get_args():
    """Arguments for the CLI"""
    parser = argparse.ArgumentParser(description='Process biophysical variables from 57.5° vegetation masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', metavar='I', type=str, required=True,
                        help='Input directory path or input mask', dest='input')
    parser.add_argument('-l', '--focal-length', metavar='FOCAL', type=float, required=True,
                        help='Focal length (mm)', dest='focallength')
    parser.add_argument('-W', '--sensor-width', metavar='SENSOR_WIDTH', type=float, required=True,
                        help='Image sensor width (mm)', dest='sensorwidth')
    parser.add_argument('-H', '--sensor-height', metavar='SENSOR_HEIGHT', type=float, required=True,
                        help='Image sensor height (mm)', dest='sensorheight')
    parser.add_argument('-f', '--fov', metavar='FOV', type=float, default=10,
                        help='This allows extracting the useful part of the images, i.e, 0° ± fov° to compute the cover fraction at 57°. (°)', dest='fov')
    parser.add_argument('--veg-color', metavar='VEG', type=int, default=255,
                        help='The vegetation color in the mask image [0-255]', dest='veg')
    parser.add_argument('--sky-color', metavar='SKY', type=int, default=0,
                        help='The sky color in the mask image [0-255]', dest='sky')
    parser.add_argument('--gap-color', metavar='GAP', type=int, default=100,
                        help='The gap color in the mask image [0-255]', dest='gap')
    parser.add_argument('-c', '--cell-size', metavar='CELL_SIZE', type=int, default=30,
                        help='The side size of a square-shaped cell (pxl)', dest='cellsize')
    parser.add_argument('-n', '--no-correction', action='store_true', default=False,
                        help='If True no apply optical correction and FOV due to the lens')
    parser.add_argument('-o','--output-format', metavar='FORMAT', type=str, default='txt',
                        help="The output format ('txt' or 'html')", dest='format')

    return parser.parse_args()

if __name__ == '__main__':
    """Main function for CLI"""
    args = get_args()
    
    #Accepted image extensions
    img_extensions = ['jpg','jpeg','png','tif','tiff']
    DECIMAL = 2
    
    #Inputs
    input_masks = []
    if os.path.isdir(args.input):
        for ext in img_extensions:
            input_masks += glob.glob(os.path.join(args.input,f'*.{ext}'), recursive=False)
    else:
        input_masks = [args.input]
        
    if len(input_masks)==0:
        print("No input image found.")
        sys.exit()
        
    if str.lower(args.format)=='html':
        extension='.html'
    else:
        extension='.txt'
        
    #Get Camera model & Image size
    model="Unknown"
    size=(0,0)

    tmp = Image.open(input_masks[0])
    if 272 in tmp.getexif().keys():
        model = tmp.getexif()[272]
    size = tmp.size
    del tmp
    
    #Set output path
    if os.path.isdir(args.input):
        dir_name = os.path.split(args.input)[1] if len(os.path.basename(args.input))!=0 else os.path.split(os.path.split(args.input)[0])[1]
        output_path = os.path.join(args.input,f"P57_{dir_name}_output{extension}")
    else:
        output_path = os.path.join(os.path.dirname(os.path.relpath(args.input)),f"P57_{os.path.splitext(os.path.basename(args.input))[0]}_output{extension}")
    
    #Write output header
    with open(output_path,'w') as output:
        
        if not args.no_correction:
            sensor_size = (args.sensorwidth,args.sensorheight)
            focal = args.focallength
            fov = args.fov
        else:
            sensor_size = "No optical correction"
            focal = "No optical correction"
            fov = "No optical correction"
        
        #Compute the biophysical variables for each input images
        res = np.zeros((len(input_masks),7))
        with tqdm(total=len(input_masks)) as pbar:
            for i in range(len(input_masks)):
            
                dp=DP57(input_masks[i], vegColor=args.veg, skyColor=args.sky, gapColor=args.gap)
                if not args.no_correction:
                    dp.param(focal_length=args.focallength, img_sensor_w=args.sensorwidth, img_sensor_h=args.sensorheight, fov=args.fov, cell_size=args.cellsize)
                else:
                    dp.param(focal_length=0, img_sensor_w=0, img_sensor_h=0, fov=0, cell_size=args.cellsize)
                
                res[i,0] = dp.P0()
                if res[i,0]==0:
                    res[i,1:]=np.nan
                    continue
                res[i,1] = dp.PAI57()
                res[i,2] = dp.PAItrue(PAIsat=8)
                res[i,3] = dp.PAItrue(PAIsat=9)
                res[i,4] = dp.PAItrue(PAIsat=10)
                res[i,5] = dp.PAItrue(PAIsat=11)
                res[i,6] = dp.PAItrue(PAIsat=12)
                
                pbar.update()
                
    if str.lower(args.format)=="html":
        res_repr = '\n'.join(
                            f'<tr><td style="font-weight:bold; font-size:small; font-family:initial;">{input_masks[i]}</td><td  style="font-family:initial;" align="center">' \
                                + '</td><td style="font-family:initial;" align="center">'.join(f'%0.{DECIMAL}f' %x for x in res[i]) \
                                + '</td></tr>' \
                            for i in range(len(res)))
        average_repr = '<tr><td style="font-family:initial;" align="center">' \
                        + '</td><td style="font-family:initial;" align="center">'.join(f'%0.{DECIMAL}f' %x for x in np.mean(res,axis=0)) \
                        + '</td></tr>'
        
    else:
        #txt format
        res_repr = '\n'.join(f"{input_masks[i]},\t"+ ',\t'.join(f'%0.{DECIMAL}f' %x for x in res[i]) for i in range(len(res)))
        average_repr = np.array2string(np.mean(res,axis=0),precision=DECIMAL,separator=",\t")[1:-1]

    d = {
        'input': args.input,
        'date': date.today(),
        'model': model,
        'veg': args.veg,
        'sky': args.sky,
        'gap': args.gap,
        'img_size': size,
        'sensor_size': sensor_size,
        'focal': focal,
        'fov': fov,
        'cell_size': args.cellsize,
        'res': res_repr,
        'sky_classif': round(np.mean(res[:,0])*100,DECIMAL),
        'green_classif': round(100-np.mean(res[:,0])*100,DECIMAL),
        'average': average_repr
    }

    with open(f"template_pai57/template{extension}", 'r') as template_file:
        template = Template(template_file.read())
        output_content = template.substitute(d)
        
        with open(output_path,'w') as output:
            output.write(output_content)
        
    print(f"Output file printed to '{output_path}'.")
