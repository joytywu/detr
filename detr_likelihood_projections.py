import pathlib as plb
import nibabel as nib
import numpy as np
import sys, os
import glob
import json
from tqdm import tqdm
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
import torch, torchvision
import torchvision.transforms as T


# adapted this so doesn't need to crop/reschale again.
def create_mipNIFTI_from_3D(img, nb_image=48):
    ls_mip=[]
    
    img_data=img.get_fdata()
    shape = img_data.shape
    print(shape)
    diag = int(np.ceil(np.sqrt(np.square(shape[0])+np.square(shape[1]))))
    max_width = max([450, diag])
    target_shape = (shape[2], max_width) # (y_, x_)
    print(target_shape)

    #Modified nifti header saving useful axial slices information
    # Can't seem to create new fields but can use existing fields to store other information...
    header = img.header.copy()
    liver_idx = img_data.shape[-1]//2
    suv_liver = img_data[:,:,liver_idx].squeeze().max()
    suv_brain = img_data[:,:,-1].squeeze().max()
#     print('Liver SUV max', suv_liver)
    header['intent_p1'] = suv_liver
    header['intent_p2'] = suv_brain
    header['intent_p3'] = img_data.max()
    # Can't store too many letters...
    header['intent_name'] = b'liver;brain;max'
    
#     print('Interpolating')
    img_data+=1e-5
    for angle in tqdm(np.linspace(0,360,nb_image)):
        #ls_slice=[]
        # This step is slow: https://stackoverflow.com/questions/14163211/rotation-in-python-with-ndimage
#         vol_angle= scipy.ndimage.interpolation.rotate(img_data,angle,order=0)
        vol_angle = scipy.ndimage.rotate(img_data,angle,order=0)
        
        MIP=np.amax(vol_angle,axis=1)
        MIP-=1e-5
        MIP[MIP<1e-5]=0
        MIP=np.flipud(MIP.T)
        MIP=to_shape(MIP, target_shape)
        ls_mip.append(MIP)
#         print('angle:', angle, MIP.shape)
    
    new_data = np.dstack(ls_mip) #shape [:,:,i]
    mip_nifti = nib.Nifti1Image(new_data, None, header)
    
    return mip_nifti

# Pad to same shape before stack
def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    x_pad = abs((x_-x))
    y_pad = abs((y_-y)) # should be 0
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2),
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant') #Default is 0


def plot_finetuned_results(pil_img, prob=None, boxes=None, labels = True):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if prob is not None and boxes is not None:
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))   
            if labels:
                cl = p.argmax()
                #print('#', cl,'##', p)
                text = f'{finetuned_classes[cl]}: {p[cl]:0.2f}'
                ax.text(xmin, ymin, text, fontsize=8,
                      bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def _suv_to_greyPIL(suv_img, suv_max):
    # Assuming eval mode
    norm = plt.Normalize(vmin = 0, vmax = suv_max) # normally radiologists view images at this suv norm
    # Color map to gray images, output has 3 channels
    cmap = plt.cm.Greys
    img = cmap(norm(suv_img))[:,:,:3].copy() # drop the alpha channel that we don't need. has shape (H x W x 3C) 
    #img = torch.as_tensor(img, dtype=torch.float64) # the transforms latera will turn the numpy array to a tensor (C x H x W)
    img = Image.fromarray(np.uint8(img*255), 'RGB') # expected input by transforms
    return img

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def filter_bboxes_from_outputs(outputs, size, threshold=0.7):
  
    # keep only predictions with confidence above threshold
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    probas_to_keep = probas[keep]

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], size)

    return probas_to_keep, bboxes_scaled


def get_all_predicted_bboxes(mip_stack, model, suv_max = 6, angles = 48, labels = True, threshold = 0.5):
    # Collate predicted bboxes
    bboxes_dict = {}
    for angle in tqdm(range(angles)):
        suv_img = mip_stack[:,:,angle].copy()
        # numpy of pet in suv values
        PIL_image = _suv_to_greyPIL(suv_img, suv_max)
        size = PIL_image.size
        # mean-std normalize the input image (batch-size: 1)
        img = transform(PIL_image).unsqueeze(0)
        # propagate through the model
        outputs = model(img)
        #print('Nth angle:', angle)
        probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs, size, threshold=threshold)
        #print(bboxes_scaled)
        bboxes_dict[angle] = {'boxes':bboxes_scaled.detach().numpy(), # tensor([x_min,y_min,x_max,y_max],[]..)
                              'likelihood':probas_to_keep.detach().numpy()}
        
        #plot_finetuned_results(PIL_image, probas_to_keep, bboxes_scaled, labels = labels)
        
    return bboxes_dict


def get_projected_bbox_likelihood(bboxes_dict, mip_stack, pet_img):
    height, width, num_angles = mip_stack.shape
    
    # the width that we will cut down to
    # cut_width = int(width/np.sqrt(2))
    cut_width = pet_img.shape[0]

    together = np.zeros((cut_width, cut_width, height))

    for i in tqdm(range(num_angles)):
        angle = 360*i/num_angles
        boxes = bboxes_dict[i]['boxes']
        box_probs = bboxes_dict[i]['likelihood']

        flat = np.zeros((height, width))
        for j in range(len(boxes)):
            # if max(box_probs[j]) < 0.05: continue           
            xmin, ymin, xmax, ymax = boxes[j]
            xmin = int(xmin)
            xmax = int(xmax)
            ymin = int(ymin)
            ymax = int(ymax)
            flat[ymin:ymax, xmin:xmax] = 1
        #plt.imshow(flat,cmap='Greys')

        # Turn the flat 2D image into 3D image
        fat = np.tile(flat, (width, 1, 1))
        # reorder axes for rotation
        fat = np.transpose(fat, axes=(0,2,1))
        # rotate back
        rot = scipy.ndimage.rotate(fat,-angle,order=0) 
        print(rot.shape)
        # size of the rotated image, which depends on angle
        w = rot.shape[0]
        assert w >= cut_width
        a = (w-cut_width)//2
        b = a + cut_width
        cut = rot[a:b, a:b, :]
        print(cut.shape)
        print()
        together = together + cut
    
    # Reverse the MIP transformations
    # flip whole stack upside down again
    reoriented = np.flipud(together).copy()
    # And rotate back to original orientation
    reoriented = np.rot90(reoriented, k=3) 
    
    return reoriented


def load_model(num_classes, model_path):
    model = torch.hub.load('facebookresearch/detr',
                       'detr_resnet50',
                       pretrained=False,
                       num_classes=num_classes)

    checkpoint = torch.load(model_path, map_location='cpu')

    model.load_state_dict(checkpoint['model'],
                          strict=False)
    
    return model.eval()


def run_workflow(data_in_root, data_out_root, model_path, transform, finetuned_classes):
    num_classes = len(finetuned_classes)
    
    root = plb.Path(data_in_root/plb.Path('petmr_detr_dataset/test/SUV_MIP/'))
    study_paths = list(root.glob('*.nii.gz'))
    study_paths = [x for x in study_paths if 'suv' in str(x).lower()]
    study_names = [str(x.name) for x in study_paths] # turns path back to str data type
    pet_paths = [os.path.join('/master/image_for_train_processed',name) for name in study_names]
    pet_paths = [str(data_in_root)+p for p in pet_paths] #somehow path won't add any other way...
     
    model = load_model(num_classes, model_path)
    
    for mip_path, pet_path in tqdm(zip(study_paths, pet_paths)):
        print('Processing study:', mip_path)
        pet = nib.load(pet_path)
        pet_img = pet.get_fdata()
        header = pet.header
        mip_pet = nib.load(mip_path)
        mip_stack = mip_pet.get_fdata()
        num_angles = mip_stack.shape[2]
        print('Tumor bbox objects detection')
        bboxes_dict = get_all_predicted_bboxes(mip_stack, model, suv_max = 6, angles = num_angles, threshold = 0.7)
        print('Generating 3D Bbox likelihood back projections')
        bbox_likelihoods_3D = get_projected_bbox_likelihood(bboxes_dict, mip_stack, pet_img)
        likelihood_nifti = nib.Nifti1Image(bbox_likelihoods_3D, None, header)
        print('Saving results')
        nii_outpath = os.path.join(data_out_root, 'NIFTI', mip_path.name)
        nib.save(likelihood_nifti, nii_outpath)
        json_outpath = os.path.join(data_out_root, 'JSON', '{}.json'.format(mip_path.name))
        with open(json_outpath,"w") as outfile:
            json.dump(bboxes_dict, outfile, cls=NpEncoder)
            print('Saved results:', mip_path)
        

# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj) 


if __name__ == "__main__":
    data_in_root = plb.Path(sys.argv[1])  # path to parent directory for all studies, e.g. /gpfs/fs0/data/stanford_data/
    data_out_root = plb.Path(sys.argv[2])  # path to where we want to save the DETR dataset, e.g. /gpfs/fs0/data/stanford_data/petmr_detr_dataset/test/
    if not os.path.isdir(data_out_root):
        os.makedirs(data_out_root)
        os.makedirs(os.path.join(data_out_root,'NIFTI'))
        os.makedirs(os.path.join(data_out_root,'JSON'))
    model_path = plb.Path(sys.argv[3]) # path to trained model checkpoint, # /home/joywu/detr/outputs/baseline/checkpoint.pth
    
    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # number of finetuned classes
    finetuned_classes = [
          'N/A', 'tumor',]
    
    run_workflow(data_in_root, data_out_root, model_path, transform, finetuned_classes) 
    
    # python detr_likelihood_projections.py /gpfs/fs0/data/stanford_data/ /gpfs/fs0/data/stanford_data/petmr_detr_dataset/test/bbox_likelihood_projections/ /home/joywu/detr/outputs/baseline/checkpoint.pth
    
    
    