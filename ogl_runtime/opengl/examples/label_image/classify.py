# -*- coding: utf-8 -*-

from torchvision import transforms as T
import torchvision.models as models

from PIL import Image


model = models.resnet50(pretrained=True)
model.eval()
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), normalize])


image = Image.open('./cat.jpg')

image = transform(image).unsqueeze(0)
output_prob = model(image)

max_id = output_prob.argmax()
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]
print('output label:', classes[output_prob.argmax()])
