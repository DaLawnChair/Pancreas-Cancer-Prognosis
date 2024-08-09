import SimpleITK as sitk
import os

def convertMhaToNrrdForPostTreatment(srcPath,destPath):
    for folder in os.listdir(srcPath):
        print(folder)
         
        for file in os.listdir(f'{srcPath}/{folder}'):
            
            #Run only on mhd files, ignore the raw
            if file[-3:]=='mhd' or file[-3:]=='mha':
                mhd_path =  f'{srcPath}{folder}/{file}'

                id = folder
            
                if 'volume' in file or 'WHOLE' in file:
                    nii_path = f'{srcPath}{id+"_"+file[:-3]}'+'nrrd'
                else:
                    nii_path = f'{destPath}{id+"_"+file[:-3]}'+'seg.nrrd'
                img = sitk.ReadImage(mhd_path)           
                print(nii_path)
                #Write the mhd image as an nii image with the path of the patient 
                sitk.WriteImage(img, nii_path)

"""
#Convert the post-treatment segments into nrrd files into the destPath 
srcPath = 'PDAC-Response/PDAC-Response/ImagingData/Post-treatment/'
destPath = 'PDAC-Response/PDAC-Response/ImagingData/postHolder/'
convertMhaToNrrdForPostTreatment(srcPath,destPath) 
"""

import os
import pandas as pd 
import shutil

columns = ['CaseID_PostNAT','SubstituteIDs_forMRN','CaseID_PreNAT','TAPS_CaseIDs_PreNAT']
data = pd.read_excel('PDAC-Response_CaseIDs_Matched.xlsx', header=None, names = columns)

postData = pd.Series(data['CaseID_PostNAT']).to_list()[2:]
postData = [str(i) for i in postData]
preData = pd.Series(data['TAPS_CaseIDs_PreNAT']).to_list()[2:]


preDataPath = 'PDAC-Response/PDAC-Response/ImagingData/Pre-treatment/'
postDataPath = 'PDAC-Response/PDAC-Response/ImagingData/postHolder/'


#Link the actual images of the post segment to the postData list
postIdToPostFile = dict(zip(postData,[None]*len(postData)))
for file in os.listdir(postDataPath):
    # print(file)
    idName = file.split('_')[0]
    if idName in postIdToPostFile:
        postIdToPostFile[idName] = file

# for key, value in postIdToPostFile.items():
#     print(key, value)


# # View which ids are not in the post data
# postFileIDs = [file.split('_')[0] for file in os.listdir(postDataPath)]
# # print(os.listdir('PDAC-Response/PDAC-Response/ImagingData/Post-treatment/'))
# print( set(postIdToPostFile.keys()) - set(postFileIDs) - isActuallyNoSegmentation)
# invalids = {'100065', '546175', '100081', '100096', '546142', '546228', '100025', '546141', '546180', '100034', '546165'}
# 11 total; 5 cases that do not exist and 6 cases that exist as empty folders


# Make the pre and post data into a dictionary
postData = list(postIdToPostFile.values())
preToPost = dict(zip(preData,postData))

counter=0
for i in range(len(postData)):
    if postData[i]==None:
        counter+=1 
print(f'Number of Nones in postData: {counter}')


noneCount = 0
"""
# Copy over the post segmentations to the pre-treatment folder
for folder in os.listdir(preDataPath):
    if 'empty' in folder:
        print(f'{folder} is empty')
        continue
    postSegment = str(preToPost[folder])
    if postSegment != 'None':
        # print(postDataPath+postSegment)
        noneCount+=1 
        shutil.copy(f'{postDataPath}{postSegment}', f'{preDataPath}{folder}/{postSegment}')


print(f'# of images from predata that exist: {len(os.listdir(preDataPath))}')
print(f'# of images from predata: {len(preData)}')
print(f'# of images from pre and post that are usuable: {noneCount}')
print(f'Note we lost 11 images because 10 didn\'t have post images and 1 of the pre-treatments was empty')
"""

## Now cull the cases that don't have the pre and post treatment images:
removeIds = []
for key,value in preToPost.items():
    if value == None:
        removeIds.append(key)
        print(f'{key} will be removed')

print(f'ids to remove: {removeIds}')
for id in removeIds:
    print(f'{preDataPath}{id}')
    if os.path.exists(f'{preDataPath}{id}/'):
        shutil.rmtree(f'{preDataPath}{id}/')
    else:
        print(f'{id} does not exist in the actual file system')
    del preToPost[id]


# for key,value in preToPost.items():
#     print(f'{key} == {value}')
print(f'# of cases used in preToPost dataset: {len(preToPost)}') # Should be 86, since 6 post segmentations were empty and 1 pre-treatment was empty, so 93-7 = 86

# I DON'T THINK YOU CAN TRUST IT, JUST GO BY THE # OF IMAGES AFTER REMOVAL 

for folder in os.listdir(preDataPath):
    print(len(os.listdir(preDataPath+folder)))







