import os

from dotenv import load_dotenv

from sentiment_analysis.etap1_reddit_scrapper import RedditScraper
from sentiment_analysis.etap1_1_sentiment_analysis_preprocessing import SentimentAnalysisPreprocessor

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

    scraper = RedditScraper(client_id, client_secret, user_agent, subreddits, num_posts=1000)
    scraper.scrape(output_file="csv_files/reddit_posts_20230503.csv")
    #
    # preprocessor = SentimentAnalysisPreprocessor(
    #     input_file='csv_files/reddit_posts_20230503.csv',
    #     output_file='csv_files/preprocessed_data_20230503.csv',
    #     num_clusters=3
    # )
    # preprocessor.preprocess()
