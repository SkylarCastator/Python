"""

Useful links:
    https://github.com/aziezahmed/podcast-player/blob/master/podcast_player/podcast_player.py
"""
import os
import sys
from os.path import expanduser
import feedparser
import listparser

player = "mpv --no-video"

def get_newest_episode(podcast):
    feed = feedparser.parse(podcast)
    entry = feed.entries[0]
    return entry

def get_episode_audio_url(podcast_entry):
    links = podcast_entry["links"]
    for link in links:
        if "audio" in link["type"]:
            return link["href"]

def play_podcast_episode(url):
    os.system('clear')
    os.system(player + " " + url)

def play_podcast():
    podcast_rss_url = "https://www.npr.org/rss/podcast.php?id=510289"
    episode_info = get_newest_episode(podcast_rss_url)
    episode_url = get_episode_audio_url(episode_info)
    play_podcast_episode(episode_url)

#play_podcast()

