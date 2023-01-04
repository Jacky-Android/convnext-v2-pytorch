import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import convnext_base as create_model
import os
from pandas.core.frame import DataFrame
paths = os.listdir(r'tests')

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    num_classes = 12
    img_size =224
    data_transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    clas = []
    pathss = []
    model = create_model(num_classes=num_classes).to(device)
    model_weight_path = "./weights/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    for i in tqdm(range(len(paths))):
        img_path = "/kaggle/input/cats-12-end/cat_12_test/"+paths[i]
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path).convert('RGB')
        pathss.append(paths[i])
    
    # [N, C, H, W]
        img = data_transform(img)
    # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

    # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            class_indict = json.load(f)

    # create model
        
        # load model weights
        
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())

        #print("class: {}   prob: {:.3}".format(class_indict[str(list(predict.numpy()).index(predict.numpy().max()))],predict.numpy().max()))
        clas.append(class_indict[str(list(predict.numpy()).index(predict.numpy().max()))][:-1])
    c={"a" : pathss,"b" : clas}#将列表a，b转换成字典
    data=DataFrame(c)#将字典转换成为数据框
    outputpath='results.csv'
    data.to_csv(outputpath,sep=',',index=False,header=False)
if __name__ == '__main__':
    main()