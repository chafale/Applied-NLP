from selenium import webdriver
from bs4 import BeautifulSoup as bs
import time
import pandas as pd
import re as re

try:
    f = open("linkedin_credentials.txt", "r")
    contents = f.read()
    username = contents.replace("=", ",").split(",")[1]
    password = contents.replace("=", ",").split(",")[3]
except:
    f = open("linkedin_credentials.txt", "w+")
    username = "XXXXXXX"
    password = "XXXXXXX"
    f.write("username={}, password={}".format(username, password))
    f.close()

browser = webdriver.Chrome(executable_path="./chromedriver")
browser.maximize_window()
browser.implicitly_wait(20)
browser.get("https://www.linkedin.com/login?fromSignIn=true&trk=guest_homepage-basic_nav-header-signin")
elementID = browser.find_element_by_id('username')
elementID.send_keys(username)

elementID = browser.find_element_by_id('password')
elementID.send_keys(password)
elementID.submit()

"""
Hash-tag List :

layoffs
hiringfreeze
recession2022
womenequality
womensafety
protests
racism
"""

"""
NEGATIVE TAGS :
 protests, 
 racism, 
 layoffs, 
 depression, 
 sad, 
 layoff, 
 recession2022,
 anxiety, 
 protest, 
 grief, 
 rejection, 
 painpoint, 
 unemployment, 
 hiringfreeze
"""
HASH_TAG = "hiringfreeze"

page = f"https://www.linkedin.com/feed/hashtag/{HASH_TAG}/"
browser.get(page)
SCROLL_PAUSE_TIME = 30

# Get scroll height
last_height = browser.execute_script("return document.body.scrollHeight")

while True:
    # Scroll down to bottom
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)

    # Calculate new scroll height and compare with last scroll height
    new_height = browser.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

post_page = browser.page_source

# Use Beautiful Soup to get access tags
linkedin_soup = bs(post_page, 'html.parser')
linkedin_soup.prettify()

# Find the post blocks
containers = linkedin_soup.findAll("div", {"class": "ember-view occludable-update"})

post_dates = []
post_texts = []
post_likes = []
post_comments = []
video_views = []
media_links = []
media_type = []

# Looping through the posts and appending them to the lists


for container in containers:

    # Try function to make sure its a post and not a promotion
    try:
        posted_date = container.find("span", {"class": "visually-hidden"})
        text_box = container.find("div", {"class": "update-components-text relative feed-shared-update-v2__commentary"})
        text = text_box.find("span", {"dir": "ltr"})
        new_likes = container.findAll("li", {
            "class": "social-details-social-counts__reactions social-details-social-counts__item"})
        new_comments = container.findAll("li", {
            "class": "social-details-social-counts__comments social-details-social-counts__item"})

        # Appending date and text to lists
        post_dates.append(posted_date.text.strip())
        post_texts.append(text_box.text.strip())

        # Determining media type and collecting relevant info for each type

        # Appending likes and comments if they exist
        try:
            post_likes.append(new_likes[0].text.strip())
        except:
            post_likes.append(0)
            pass

        try:
            post_comments.append(new_comments[0].text.strip())
        except:
            post_comments.append(0)
            pass

    except:
        pass

with open(f"./hashtags/11_20_{HASH_TAG}.txt", 'w', encoding='utf-8') as f:
    print("Tag : ", HASH_TAG)
    print("Writing the file . . .")
    for line in post_texts:
        f.write(f"{line}\n")
        f.write("-----------------------------\n")
    print("Writing Complete . . . !")
