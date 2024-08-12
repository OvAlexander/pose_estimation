import pafy
import yt_dlp
import os
import re

def find(name, path):
    name = name + ".mp4"
    for root, dirs, files in os.walk(path):
        if name in files:
            return True
    return False

def fetch(uid:str):
    pattern = r"\[.{11}\]"
    url   = f"https://www.youtube.com/watch?v={uid}"
    ydl_opts = {}
    if find(uid,"./mv_videos") == False:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_mv = info["requested_downloads"][0]["filename"]
            regex_mv = re.search(pattern,downloaded_mv)
            matched_mv_name = regex_mv.group(0)
            new_mv_file_name = matched_mv_name[1:12] + ".mp4"
            os.rename(f"./{downloaded_mv}", f"./mv_videos/{new_mv_file_name}")
