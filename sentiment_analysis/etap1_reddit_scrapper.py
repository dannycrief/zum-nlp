import praw
import pandas as pd
import logging


class RedditScraper:
    def __init__(self, client_id, client_secret, user_agent, subreddits, num_posts=100):
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        for logger_name in ("praw", "prawcore"):
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)

        self.reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
        self.subreddits = subreddits
        self.num_posts = num_posts

    def scrape(self, output_file='csv_files/reddit_posts.csv'):
        posts = []
        i, j = 0, 0

        for subreddit in self.subreddits:
            i += 1
            for submission in self.reddit.subreddit(subreddit).top(limit=self.num_posts, time_filter='all'):
                if submission.is_self:
                    posts.append({
                        'author': submission.author.name if submission.author else '',
                        'title': submission.title,
                        'score': submission.score,
                        'id': submission.id,
                        'url': submission.url,
                        'num_comments': submission.num_comments,
                        'created': pd.to_datetime(submission.created_utc, unit='s'),
                        'subreddit': subreddit,
                        'selftext': submission.selftext
                    })
                    j += 1
                    print(f"INFO: Parsed i: {i} and j: {j}")

        df = pd.DataFrame(posts)
        df.to_csv(output_file, index=False)
