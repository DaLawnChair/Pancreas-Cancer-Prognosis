import slicer
import vtk
import vtkSegmentationCorePython as vtkSegmentationCore

# C:\Users\johnz\AppData\Local\slicer.org\Slicer 5.6.2\Slicer.exe --python-script D:\SimpsonLab\groupSegmentations.py

# from SegmentEditorEffects import * 
# path = '/mnt/d/SimpsonLab/pancreas_data/pancreas_data/neoadjuvant_pdac/01_neo_pdac_pre_Tumor.nii.gz'

# Get the MRML scene
scene = slicer.mrmlScene

# Get all segmentation nodes in the scene
segmentation_nodes = [scene.GetNthNode(i) for i in range(scene.GetNumberOfNodes()) if scene.GetNthNode(i).IsA('vtkMRMLSegmentationNode')]
volume_node = [scene.GetNthNode(i) for i in range(scene.GetNumberOfNodes()) if scene.GetNthNode(i).IsA('vtkMRMLScalarVolumeNode')]

print(volume_node[0].GetName())

# combinedSegmentation = slicer.mrmlScene.AddNewNodeByClass('vtkSegment')
# combinedSegmentation.SetName('Combined')


# Print all segmentation nodes
for node in segmentation_nodes:
    print(f'Segmentation Node Name: {node.GetName()}')
    # node.GetSegmentation().AddEmptySegment('Combined')

    # # Turn off the visibility of the segment that casts an overlay over the entire image
    # nodeSegmentaitonDisplay = node.GetDisplayNode()
    # segmentOverlayID = node.GetSegmentation().GetSegmentIdBySegmentName('Segment_-1000')
    # nodeSegmentaitonDisplay.SetSegmentVisibility(segmentOverlayID, False)


    # Remove segment_-1000
    node.GetSegmentation().RemoveSegment(node.GetSegmentation().GetSegmentIdBySegmentName('Segment_-1000'))


    # Add a segment called 'Combined' that will be the union of all other segments of the segmentation
    oldCombined = node.GetSegmentation().GetSegmentIdBySegmentName('Combined')
    if node.GetSegmentation().GetSegmentIdBySegmentName('Combined')!='':
        node.GetSegmentation().RemoveSegment(oldCombined)
    combinedNodeID = node.GetSegmentation().AddEmptySegment('Combined')


    print(f'Generating combinedNode: {combinedNodeID}')

    # # Create and configure the SegmentEditorWidget, which specifies the segmentation to perform and where
    # segmentationEditorWidget = slicer.qMRMLSegmentEditorWidget()
    # segmentationEditorWidget.setMRMLScene(slicer.mrmlScene)
    # segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentEditorNode')
    # segmentationEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    # segmentationEditorWidget.setSegmentationNode(node) # or node?
    # segmentationEditorWidget.setSourceVolumeNode(volume_node[0])
    
    # print(f'Attaching Editor to the segment and volume')


#     segmentationEditorWidget.setActiveEffectByName('Logical operators')

#     effect = segmentationEditorWidget.activeEffect()    
#     # print(type(effect))
#     effect.setParameter('BypassMasking', 0 )
#     effect.setParameter('Operation','FILL')
#     effect.setParameter('ModifierSegmentID',node.GetSegmentation().GetSegmentIdBySegmentName('Combined')) # combinedNodeID

#     segmentEditorNode.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone) #OverwriteVisibleSegments 
#     segmentEditorNode.SetMaskMode(slicer.vtkMRMLSegmentationNode.EditAllowedInsideVisibleSegments)
#     effect.self().onApply()
#     print(f'Applying affects')


#     segmentEditorWidget = None
#     slicer.mrmlScene.RemoveNode(segmentEditorNode)

#     # numOfSegments = node.GetNumberOfSegments().GetSedgmentIds()


#     break

#     # Turn off all nodes except for combined
# nodeSegmentaitonDisplay = node.GetDisplayNode()
# for segmentName in node.GetSegmentation().GetSegmentIDs():

#     if segmentName != 'Combined':
#         segmentOverlayID = node.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
#         nodeSegmentaitonDisplay.SetSegmentVisibility(segmentOverlayID, False)

    


# tmp = slicer.util.loadLabelVolume('/mnt/d/SimpsonLab/pancreas_data/pancreas_data/neoadjuvant_pdac/01_neo_pdac_pre/01_neo_pdac_pre_panc_segmented.nrrd')
# mask = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
# slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(tmp, mask)
# slicer.mrmlScene.RemoveNode(tmp)