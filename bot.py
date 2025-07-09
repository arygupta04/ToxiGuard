import discord
from dotenv import load_dotenv
from toxicity_wrapper import ToxicityClassifier

# This function loads environment variables from .env
load_dotenv()

TOKEN = 'MTMyNjQzMTc1ODc1Njg3MjIzNA.GNnUcn.mtEQCyMNP2SjqqXweIoLifKrcb2cZxH5-a5dTQ'
MAX_FEATURES = 200000 # number of words in the vocab


# Create an Intents object with the necessary permissions
intents = discord.Intents.default()
intents.messages = True  # Enable event handling for messages
intents.message_content = True  # Allow access to message content

client = discord.Client(intents=intents)

# Initialize the ToxicityClassifier with model path
model_path = 'best_model.h5'

classifier = ToxicityClassifier(model_path=model_path)


# This event is triggered when the bot successfully connects to Discord
@client.event
async def on_ready():
    # client.user refers to the bot
    print(f'{client.user} has connected to Discord!')

    # This refers to the general channel
    channel_id = 1326432814593343613
    channel = client.get_channel(channel_id)

    if channel:
        await channel.send("Hello! I'm monitoring for inappropriate comments.")

# This event is triggered when a message is sent in a channel the bot can access
@client.event
async def on_message(message):
    # Prevent the bot from responding to its own messages
    if message.author == client.user:
        return

    feedback = classifier.classify(message.content)
    if feedback:
        await message.delete()

        await message.author.send(
            f"Your message in `{message.channel.name}` was flagged as: {', '.join(feedback)}. "
            "Please adhere to the community guidelines and maintain a positive environment."
        )
        
client.run(TOKEN)
