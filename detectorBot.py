# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 02:49:42 2023

@author: ssour
"""

from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.AdaptiveAvgPool2d((6, 6))
                                     )
        
        self.classifier = nn.Sequential(nn.Linear(256 * 6 * 6, 4096),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, 4096),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, 2)
                                       )


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 6 * 6 * 256)
        x = self.classifier(x)
        return x

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = torch.load('melanoma_CNN.pt')
model = model.to(device)
model.eval()  # Set the model to evaluation mode



# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Send me an image and I will check it for melanoma!')

def check_melanoma(update: Update, context: CallbackContext) -> None:
    file = context.bot.getFile(update.message.photo[-1].file_id)

    # Download the photo
    file.download('input_image.jpg')

    # Open the image file
    image = Image.open('input_image.jpg')

    # Preprocess the image
    image = transform(image)

    # Unsqueeze to add a batch dimension
    image_tensor = image.unsqueeze(0)

    # Move the tensor to the same device as the model
    image_tensor = image_tensor.to(device)

    # Pass the image to the model and get the prediction
    output = model(image_tensor)
    _, predicted = torch.max(output.data, 1)

    if predicted.item() == 1:
        update.message.reply_text('Melanoma detected. Please consult a doctor immediately.')
    else:
        update.message.reply_text('No melanoma detected. Stay safe.')



def main() -> None:
    updater = Updater("6290244745:AAFeYugkH2JfhhMoBmS9Lp-4OEZw5HiXnHM", use_context=True)

    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo & ~Filters.command, check_melanoma))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

