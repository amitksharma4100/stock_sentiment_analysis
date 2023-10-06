from flask import Flask, request, jsonify, render_template
import pandas as pd
import json
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from waitress import serve
import plotly.express as px
import nltk
import requests
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__, template_folder='C:\\Users\\amitk\\OneDrive\\Desktop\\vercel-stock-sentiment\\templates')

def get_news(tickers):
    finviz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}

    for ticker in tickers:
        url = finviz_url + ticker
        req = Request(url=url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'})
        response = urlopen(req)
        html = BeautifulSoup(response, features="html.parser")
        news_table = html.find(id='news-table')
        news_tables[ticker] = {'table': news_table, 'company_name': html.title.text.split('-')[0].strip()}

    return news_tables

@app.route('/api/v1.0/get_news', methods=['GET'])
def get_news_endpoint():
    tickers_param = request.args.get('tickers')
    
    if not tickers_param:
        return jsonify(error='Please provide a list of tickers as a query parameter.')
    
    tickers = tickers_param.split(',')
    
    if not tickers:
        return jsonify(error='No tickers provided.')
    
    news_tables = get_news(tickers)
    news_tables_str = {ticker: str(table) for ticker, table in news_tables.items()}
    return jsonify(news_tables_str)

@app.route('/api/v1.0/parse_and_score_news', methods=['GET'])
def parse_and_score_news():
    tickers_param = request.args.get('tickers')
    
    if not tickers_param:
        return jsonify(error='Please provide a list of tickers as a query parameter.')
    
    tickers = tickers_param.split(',')
    
    if not tickers:
        return jsonify(error='No tickers provided.')
    
    news_tables = get_news(tickers)

    # Call the get_news function with the tickers list
    news_tables = get_news(tickers)
    parsed_news = []

    for ticker, data in news_tables.items():
        news_table = data['table']
        company_name = data['company_name']

        for x in news_table.findAll('tr'):
            a_element = x.a

            if a_element is not None:
                text = a_element.get_text()
                td_text = x.td.text.split()

                if len(td_text) == 2:
                    date = td_text[0]
                    time = td_text[1]
                else:
                    date = None
                    time = None

                parsed_news.append({'ticker': ticker, 'date': date, 'time': time, 'headline': text, 'company_name': company_name})

    parsed_news_df = pd.DataFrame(parsed_news)
    
    # Handle date and time formatting
    if parsed_news_df['date'].notna().any():
        parsed_news_df['datetime'] = pd.to_datetime(parsed_news_df['date'] + ' ' + parsed_news_df['time'], errors='coerce')
    else:
        parsed_news_df['datetime'] = None

    parsed_news_df.drop(columns=['date', 'time'], inplace=True)
    parsed_news_df['datetime'] = parsed_news_df['datetime'].dt.strftime('%m/%d/%Y %H:%M')

    vader = SentimentIntensityAnalyzer()
    scores = parsed_news_df['headline'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)

    parsed_and_scored_news = parsed_news_df.join(scores_df, rsuffix='_right')
    parsed_and_scored_news = parsed_and_scored_news.set_index('datetime')  # Set the datetime column as the index
    parsed_and_scored_news = parsed_and_scored_news.drop(['headline'], axis=1)
    parsed_and_scored_news = parsed_and_scored_news.rename(columns={"compound": "sentiment_score"})

    # Initialize error as an empty string
    error = ""

    # Check if the DataFrame is not empty before converting to JSON
    if not parsed_and_scored_news.empty:
        json_response = parsed_and_scored_news.reset_index().to_json(orient='records')
    else:
        json_response = '[]'  # Return an empty JSON array if the DataFrame is empty

    response = app.response_class(response=json_response, status=200, mimetype='application/json')

    start_date = request.args.get('start_date')  # Get the start date from the URL parameters
    end_date = request.args.get('end_date')      # Get the end date from the URL parameters

    # Filter the DataFrame based on the start and end dates (if provided)
    if start_date and end_date:
        start_date = pd.to_datetime(start_date, format='%Y-%m-%d', errors='coerce')
        end_date = pd.to_datetime(end_date, format='%Y-%m-%d', errors='coerce')

        if not start_date.isnull().values.any() and not end_date.isnull().values.any():
            filtered_df = parsed_and_scored_news[(parsed_and_scored_news.index >= start_date) & (parsed_and_scored_news.index <= end_date)]
        else:
            return jsonify(error='Invalid date format or range.')

        # Convert the filtered DataFrame to JSON response
        json_response = filtered_df.reset_index().to_json(orient='records')
    else:
        # If start_date and end_date are not provided, return the entire DataFrame
        json_response = parsed_and_scored_news.reset_index().to_json(orient='records')

    response = app.response_class(response=json_response, status=200, mimetype='application/json')

    return response

@app.route("/", methods=['GET', 'POST'])
def welcome():
    submitted = False
    ticker = None
    overall_average_score_str = None
    plot_html = None
    error = ""  # Initialize error as an empty string
    company_name = ""  # Initialize company_name as an empty string

    if request.method == 'POST':
        ticker = request.form.get('ticker')
        if ticker:
            response = requests.get(f'http://127.0.0.1:5000/api/v1.0/parse_and_score_news?tickers={ticker}')  # Updated query parameter name
            if response.status_code == 200:
                parsed_and_scored_news = pd.read_json(response.text)
                
                if not parsed_and_scored_news.empty:  # Check if the DataFrame is not empty
                    ticker_data = parsed_and_scored_news[parsed_and_scored_news['ticker'] == ticker]
                    
                    if not ticker_data.empty:
                        company_name = ticker_data.iloc[0]['company_name']  # Get the company name from the first row
                        mean_scores = ticker_data.resample('W', on='datetime')['sentiment_score'].mean()
                        mean_scores = mean_scores.reset_index()
                        mean_scores.columns = ['datetime', 'sentiment_score']
                        
                        fig = px.bar(mean_scores, x='datetime', y='sentiment_score', title=f'{ticker} Weekly Sentiment Scores')
                        plot_html = fig.to_html()

                        overall_average_score = ticker_data['sentiment_score'].mean()
                        overall_average_score_str = f'Overall Average Sentiment Score: {overall_average_score:.2f}'
                    else:
                        error = f'No sentiment data available for {ticker}.'
                else:
                    error = f'No data available for {ticker}.'
            else:
                error = f'Error fetching data for {ticker}.'
        else:
            error = 'Please enter a valid ticker symbol.'

        submitted = True
        print(f"Company Name: {company_name}")
    return render_template('index4.html', submitted=submitted, ticker=ticker, overall_average_score=overall_average_score_str, plot_html=plot_html, error=error, company_name=company_name)  # Pass company_name to the template

if __name__ == '__main__':

 serve(app, host='0.0.0.0', port=5000)
   
#app.run()  # This line is used for local development only