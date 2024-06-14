import SimpleITK as sitk
import os

def findNthOccurrance(val, n):
    start = val.find('_')
    while start >=0 and n>1:
        start = val.find('_',start+len('_'))
        n-=1 
    return start 

def convertToNii(root_path,maxCount=2,idenifierUnderscroreCount=4):
    for folder in os.listdir(root_path):
        print(folder)
         
        # nii_folder = folder+'_nii'
        # if os.path.isdir(nii_folder):
        #     continue 
        # os.mkdir(root_path+nii_folder)

        #Iterate over maxCount number of ids
        count = 0
        for file in os.listdir(f'{root_path}/{folder}'):
            
            #Run only on mhd files, ignore the raw
            if file[-3:]=='mhd' or file[-3:]=='mha':
                mhd_path =  f'{root_path}{folder}/{file}'

                id = file[:findNthOccurrance(file,idenifierUnderscroreCount)]
                #Make a new directory if it doesn't exist
                if not os.path.isdir(root_path+id):
                    os.mkdir(root_path+id) 
                
                if 'volume' in file or 'WHOLE' in file:
                    nii_path = f'{root_path}{id}/{file[:-3]}'+'nrrd'
                else:
                    nii_path = f'{root_path}{id}/{file[:-3]}'+'seg.nrrd'
                img = sitk.ReadImage(mhd_path)           
                #Write the mhd image as an nii image with the path of the patient 
                sitk.WriteImage(img, nii_path)

                count +=1
            if count>maxCount-1:
                break

# convertToNii('/mnt/d/SimpsonLab/PDAC response/PDAC response/ImagingData/Pre-treatment/',idenifierUnderscroreCount=1)


# convertToNii('/mnt/d/SimpsonLab/PDAC response/PDAC response/ImagingData/Pre-treatment/',idenifierUnderscroreCount=1)


root_path = 'PDAC-Response/PDAC-Response/ImagingData/Pre-treatment/'
# for folder in os.listdir(root_path):
#     os.mkdir(f'{root_path}{folder}/mha')
#     for file in os.listdir(f'{root_path}/{folder}'):
#         print(f'{root_path}{folder}/{file}')


mhd_path = root_path + 'CASE533/mha/CASE533_BASE_PRT_WHOLE_CT.mha'
nii_path = root_path + 'CASE533/mha/CASE533_BASE_PRT_WHOLE_CT.nrrd'

img = sitk.ReadImage(mhd_path)           
#Write the mhd image as an nii image with the path of the patient 
sitk.WriteImage(img, nii_path)
