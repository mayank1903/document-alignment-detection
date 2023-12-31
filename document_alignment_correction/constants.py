import torch
import torchvision.transforms as transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


label_mapping = {0:0, 90:1, 180:2, 270:3}
pred_map = {value: key for key, value in label_mapping.items()}

train_params = {"batch_size": 8,
         "num_workers": 0,
         "shuffle": True}

test_params = {"batch_size": 8,
         "num_workers": 0,
         "shuffle": False}

#prediction transformation
prediction_transform = transforms.Compose([
                transforms.ToTensor()
            ])
