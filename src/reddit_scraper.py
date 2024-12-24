import time
from bs4 import BeautifulSoup
import os
import csv 
import praw
from datetime import datetime
from config import CLIENT_ID, CLIENT_SECRET, USER_AGENT, SUBREDDIT_LIST, ALL_SUBREDDITS
def main():
    reddit = praw.Reddit(       #authenticate into reddit
        client_id= CLIENT_ID,
        client_secret= CLIENT_SECRET,
        user_agent= USER_AGENT
    )

    filename = 'comments3.csv'

    for sub in ALL_SUBREDDITS:
        platform = sub
        subreddit = reddit.subreddit(sub) #initialize subreddit
        new_posts = subreddit.top(time_filter='year', limit=150) #grab the posts


        with open(filename, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            if not os.path.isfile(filename) or os.stat(filename).st_size == 0: #if file doesn't exist, create it
                writer.writerow(['Author', 'Comment Body', 'Time', 'Platform'])


            for post in new_posts: #loop over each post
                sleep_until_reset(reddit)
                post.comments.replace_more(limit=None)  # replace "MoreComments" objects with the actual comments

                for comment in post.comments.list():
                    realtime = datetime.fromtimestamp(comment.created_utc)
                    writer.writerow([comment.author.name if comment.author else "Deleted", comment.body, comment.created_utc, platform])

            


def sleep_until_reset(reddit):
    if reddit.auth.limits['remaining'] <= 100: #when we go over api limit
        print("danger of exceeding API limit, sleeping...")
        reset_timestamp = reddit.auth.limits['reset_timestamp']
        reset_time = datetime.fromtimestamp(reset_timestamp)
        now = datetime.now()
        sleep_time = (reset_time - now).total_seconds()
        print(sleep_time)
        if sleep_time > 0:  # Sleep only if reset_time is in the future
            time.sleep(sleep_time)



if __name__ == "__main__":
    main()