import torch
from tqdm import tqdm
import numpy as np
import itertools
from PIL import Image
from torchvision import transforms
import glob
from model import FaceNet2
# --------------------gao.tv add some function ----------------
def getVector():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device is: ', device)
    model = torch.load('InceptionResNetV1_ArcFace.pt')
    model = model.to(device)

    # model =  FaceNet2(num_classes=10, device=device, pretrain=False)
    # model.load_state_dict(torch.load('..\\data\\SamYuen\\InceptionResNetV1_ArcFace.pt'))

    transforms_list2 = [transforms.Resize((128, 128)), transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    transforms_test = transforms.Compose(transforms_list2)

    allGen = glob.glob('E:\\Desktop\\samsung\\Doan\\training_mask\\*')

    with torch.no_grad():
        model.eval()
        for gen in tqdm(allGen):
            allImage = glob.glob(gen + '/*.jpg')
            allImage += glob.glob(gen + '/*.png')
            allImage += glob.glob(gen + '/*.jpeg')
            allImage += glob.glob(gen + '/*.jfif')
            for imgLink in allImage:
                img = Image.open(imgLink).convert('RGB')
                img = transforms_test(img)
                img = img.unsqueeze(0)
                test_image = img.to(device)
                test_out = model(test_image)
                np_arr = test_out['embeddings'].cpu().detach().numpy()
                print(np.array(np_arr))
                # nameFile = imgLink[:-3] + 'txt'
                # np.savetxt(nameFile, np_arr, delimiter=',', fmt='%.16f')
getVector()