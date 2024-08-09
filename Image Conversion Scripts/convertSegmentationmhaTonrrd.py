import SimpleITK as sitk
import os

def findNthOccurrance(val, n):
    start = val.find('_')
    while start >=0 and n>1:
        start = val.find('_',start+len('_'))
        n-=1 
    return start 

def convertToSingleMask(original):
    originalArray = sitk.GetArrayFromImage(original)
    originalArray[originalArray == -1000] = 0
    originalArray[originalArray > 1] = 1
    originalArray[originalArray <0] = 1
    modified = sitk.GetImageFromArray(originalArray)
    modified.CopyInformation(original)

    return modified 

def convertToNii(start_path, end_path,ignoreFolder,convertFolderImagesToMask,idenifierUnderscroreCount=4):
    for folder in os.listdir(start_path):
        if ignoreFolder in folder:
            continue
        print(folder) 

        convertToMask = False
        if folder in convertFolderImagesToMask:
            convertToMask = True 

        for file in os.listdir(f'{start_path}{folder}'):
            
            #Run only on mhd files, ignore the raw
            if file[-3:]=='mhd' or file[-3:]=='mha':
                mhd_path =  f'{start_path}{folder}/{file}'

                id = file[:findNthOccurrance(file,idenifierUnderscroreCount)]
                # id = folder
                # Make a new directory if it doesn't exist
                if not os.path.isdir(end_path+id):
                    os.mkdir(end_path+id) 
                
                if 'volume' in file or 'WHOLE' in file:
                    nii_path = f'{end_path}{id}/{file[:-3]}'+'nrrd'
                else:
                    nii_path = f'{end_path}{id}/{file[:-3]}'+'seg.nrrd'
                
                img = sitk.ReadImage(mhd_path)           
                if convertToMask:
                    img = convertToSingleMask(img)
                print(nii_path)
                #Write the mhd image as an nii image with the path of the patient 
                sitk.WriteImage(img, nii_path)


root_path = 'PDAC-Response/PDAC-Response/ImagingData/Pre-treatment/'

## For multiple cases


convertToNii('pancreas_data/pancreas_data/neoadjuvant_pdac/', 'pancreas_data/pancreas_data/neoadjuvant_pdac/', ignoreFolder='data_segmented_panc', convertFolderImagesToMask=['data_segmented_tumor'], idenifierUnderscroreCount=1)
# convertToNii('/mnt/d/SimpsonLab/PDAC response/PDAC response/ImagingData/Pre-treatment/',idenifierUnderscroreCount=1)


# for folder in os.listdir(root_path):
#     os.mkdir(f'{root_path}{folder}/mha')
#     for file in os.listdir(f'{root_path}/{folder}'):
#         print(f'{root_path}{folder}/{file}')


## For a single case
# mhd_path = root_path + 'CASE533/mha/CASE533_BASE_PRT_WHOLE_CT.mha'
# nii_path = root_path + 'CASE533/mha/CASE533_BASE_PRT_WHOLE_CT.nrrd'

# img = sitk.ReadImage(mhd_path)           
# #Write the mhd image as an nii image with the path of the patient 
# sitk.WriteImage(img, nii_path)
