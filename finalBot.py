# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 02:54:44 2023

@author: ssour
"""

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
import random
from telegram.ext import ConversationHandler





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

CHOOSE_DOCTOR, CHOOSE_TIMESLOT, SEND_LINK = range(3)

dermatologists = [
    {"name": "Dr. Sandra Lee (aka Dr. Pimple Popper)", 
     "description": "Known for her YouTube channel and expertise in dermatology and cosmetic surgery. Popular for showcasing various dermatological procedures, especially extractions of blackheads and cysts. Based in California, USA."},
     
    {"name": "Dr. Harold Lancer", 
     "description": "Highly respected dermatologist based in Beverly Hills, California. Specializes in advanced skincare treatments and anti-aging techniques. Has a celebrity clientele and a renowned skincare line, Lancer Skincare."},
     
    {"name": "Dr. Doris Day", 
     "description": "Board-certified dermatologist in New York City. Specializes in cosmetic dermatology, laser procedures, and minimally invasive facial rejuvenation techniques. Author of skincare books and a frequent guest expert on TV shows."}
]

timeslots = ['10:00 AM - 11:00 AM', '2:00 PM - 3:00 PM', '5:00 PM - 6:00 PM']



def check_melanoma(update: Update, context: CallbackContext) -> int:
    file = context.bot.getFile(update.message.photo[-1].file_id)
    file.download('input_image.jpg')
    image = Image.open('input_image.jpg')
    image = transform(image)
    image_tensor = image.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    _, predicted = torch.max(output.data, 1)

    if predicted.item() == 1:
        update.message.reply_text('Melanoma detected. Would you like to consult with one of our verified dermatologists? (y/n)')
        return CHOOSE_DOCTOR
    else:
        update.message.reply_text('No melanoma detected. Stay safe.')
        return ConversationHandler.END

def choose_doctor(update: Update, context: CallbackContext) -> int:
    user_response = update.message.text.lower()
    if user_response == 'y':
        for i, dermatologist in enumerate(dermatologists):
            update.message.reply_text(f'{i+1}. {dermatologist["name"]}: {dermatologist["description"]}')
        return CHOOSE_TIMESLOT
    else:
        update.message.reply_text('Stay safe and take care.')
        return ConversationHandler.END

def choose_timeslot(update: Update, context: CallbackContext) -> int:
    chosen_doctor_index = int(update.message.text) - 1
    if 0 <= chosen_doctor_index < len(dermatologists):
        random.shuffle(timeslots)
        context.user_data['chosen_doctor_index'] = chosen_doctor_index
        update.message.reply_text(f'Available timeslots for {dermatologists[chosen_doctor_index]["name"]} are:')
        for i, timeslot in enumerate(timeslots[:3]):
            update.message.reply_text(f'{i+1}. {timeslot}')
        return SEND_LINK
    else:
        update.message.reply_text('Invalid dermatologist number entered. Please try again.')
        return CHOOSE_DOCTOR

def send_link(update: Update, context: CallbackContext) -> None:
    chosen_timeslot_index = int(update.message.text) - 1
    if 0 <= chosen_timeslot_index < 3:
        chosen_doctor_index = context.user_data['chosen_doctor_index']
        telegram_link = "https://t.me/kabhoom17"  # Replace with your actual telegram link
        update.message.reply_text(f'Your appointment with {dermatologists[chosen_doctor_index]["name"]} at {timeslots[chosen_timeslot_index]} has been successfully scheduled. You can join the chat via this link: {telegram_link}')
    else:
        update.message.reply_text('Invalid timeslot number entered. Please try again.')
        return CHOOSE_TIMESLOT
    return ConversationHandler.END

def main() -> None:
    updater = Updater("6290244745:AAFeYugkH2JfhhMoBmS9Lp-4OEZw5HiXnHM", use_context=True)

    dp = updater.dispatcher
    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(Filters.photo & ~Filters.command, check_melanoma)],
        states={
            CHOOSE_DOCTOR: [MessageHandler(Filters.text & ~Filters.command, choose_doctor)],
            CHOOSE_TIMESLOT: [MessageHandler(Filters.text & ~Filters.command, choose_timeslot)],
            SEND_LINK: [MessageHandler(Filters.text & ~Filters.command, send_link)]
        },
        fallbacks=[MessageHandler(Filters.text | Filters.command, fallback)],
    )
    dp.add_handler(conv_handler)

    updater.start_polling()

    updater.idle()

def fallback(update: Update, context: CallbackContext) -> None:
    """Handle unexpected input during the conversation."""
    update.message.reply_text('Sorry, I did not understand that. Please send me a photo to check for melanoma.')

if __name__ == '__main__':
    main()
