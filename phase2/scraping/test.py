import os
from telethon import TelegramClient
from telethon.tl.types import MessageReactions
from datetime import datetime
import csv
import asyncio
import pytz
import nest_asyncio
nest_asyncio.apply()


api_id = int(os.getenv("API_ID"))  
api_hash = os.getenv("API_HASH")  
channel_username = 'OfficialPersiaTwiter'  


tz = pytz.UTC
start_date = tz.localize(datetime(2019, 1, 1))
end_date = tz.localize(datetime(2022, 8, 1))


laugh_emojis = ['😂', '🤣', '😁']


client = TelegramClient('session_name', api_id, api_hash)

async def scrape_laugh_messages():
    await client.start()

   
    with open('filtered_laugh_reactions.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Text', 'Top Reaction', 'Top Reaction Count', 'Views', 'Date'])

        async for msg in client.iter_messages(channel_username, offset_date=end_date, reverse=True):
            if msg.date < start_date:
                break

            
            if not msg.text:
                continue

            
            if msg.media:
                continue

            
            if msg.reply_to:
                continue

            
            reactions = msg.reactions
            if not isinstance(reactions, MessageReactions):
                continue

            
            top_reaction = None
            top_count = -1
            for reaction in reactions.results:
                if reaction.count > top_count:
                    top_count = reaction.count
                    top_reaction = reaction.reaction.emoticon

            
            if top_reaction in laugh_emojis:
                writer.writerow([
                    msg.text.replace('\n', ' ').strip(),
                    top_reaction,
                    top_count,
                    msg.views or 0,
                    msg.date.strftime("%Y-%m-%d")
                ])


if __name__ == "__main__":
    asyncio.run(scrape_laugh_messages())

