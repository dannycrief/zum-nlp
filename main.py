import os
import csv
import glob

from dotenv import load_dotenv

from sentiment_analysis.etap1_1_sentiment_analysis_preprocessing import SentimentAnalysisPreprocessor
from sentiment_analysis.etap1_reddit_scrapper import RedditScraper


def combine_tsv_files(file_list: list, output_file: str):
    post_data = [
        row for file in file_list for row in
        csv.DictReader(open(file, 'r', newline='', encoding='utf-8'), delimiter='\t')
    ]

    with open(output_file, 'w', newline='', encoding='utf-8') as o_file:
        writer = csv.DictWriter(o_file, fieldnames=post_data[0].keys(), delimiter='\t')
        writer.writeheader()
        writer.writerows(post_data)


if __name__ == '__main__':
    load_dotenv()

    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    user_agent = os.getenv('USER_AGENT')

    subreddits = [
        'worldnews', 'news', 'poland', 'ukraine', 'economics',
        'geopolitics', 'europe', 'finance', 'business', 'technology',
        'environment', 'science', 'globalhealth', 'energy', 'internationalpolitics',
        'cybersecurity', 'education', 'humanrights', 'globaldevelopment'
    ]

    scraper = RedditScraper(client_id, client_secret, user_agent, subreddits, num_posts=10000)
    # scraper.scrape(output_file="csv_files/01_reddit_posts/reddit_posts_20230503.tsv")
    # Run line above if you do not have any .tsv files

    reddit_posts = glob.glob("csv_files/01_reddit_posts/*.tsv")
    combine_tsv_files(reddit_posts, "csv_files/01_reddit_posts/reddit_posts_combined.tsv")

    preprocessor = SentimentAnalysisPreprocessor(
        input_file='csv_files/01_reddit_posts/reddit_posts_combined.tsv',
        output_file='csv_files/02_preprocessed_data/preprocessed_data.tsv'
    )
    preprocessor.preprocess()
