import telegram
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as pdr
from io import BytesIO
import os
import string
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import json
import csv
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
import warnings
from alpha_vantage.fundamentaldata import FundamentalData
from pandas import read_csv, DataFrame
import statsmodels.api as sm
from statsmodels.iolib.table import SimpleTable
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import statsmodels.tsa.api as tsa
import statsmodels.tsa.statespace as statespace
import FundamentalAnalysis as fa
warnings.filterwarnings("ignore")
input_ticker_global = ''
input_model_global = ''


def start(update, context):
	context.bot.send_message(chat_id = update.effective_chat.id, text = '''
Oh, Hello there! 
Thanks for checking out our financial bot

Please get familiar with commands of the bot to use the functionality:

/stock_info Provides current price, market state and graph
Example: /stock_info AMZN

/stock_analysis Provides most popular stock parameters as well as the rating
Example: /stock_performance AMZN

/stock_prediction Provides price predictions based on statistical models
Example: /stock_prediction AMZN

/crypto_info Provides the most general information of a cryptocurrency
Example: /crypto_info BTC

/crypto_analysis Provides the current health state of a cryptocurrency
Example: /crypto_analysis BTC

/exchange Provides the exchange rate ratio of a currency pair. (works for cryptocurrencies)
Example: /exchange BTC USD

/stock_overview

/abbreviation_list

/help Gives the list of all useful commands
''')

def help(update, context):
    context.bot.send_message(chat_id = update.effective_chat.id, text = '''
Oh, Hello there! 
Thanks for checking out our financial bot

Please get familiar with commands of the bot to use the functionality:

/stock_info Provides current price, market state and graph
Example: /stock_info AMZN

/stock_analysis Provides most popular stock parameters as well as the rating
Example: /stock_performance AMZN

/stock_prediction Provides price predictions based on statistical models
Example: /stock_prediction AMZN

/crypto_info Provides the most general information of a cryptocurrency
Example: /crypto_info BTC

/crypto_analysis Provides the current health state of a cryptocurrency
Example: /crypto_analysis BTC

/exchange Provides the exchange rate ratio of a currency pair. (works for cryptocurrencies)
Example: /exchange BTC USD

/stock_overview

/abbreviation_list
''')



def stock_info(URL):
	res_array = []
	
	page = requests.get(URL)

	soup = BeautifulSoup(page.content, 'html.parser')

	results = soup.find(id='quote-header-info')

	quote_name = results.find('h1', class_='D(ib) Fz(18px)')
	quote_price = results.find('span', class_ = 'Trsdu(0.3s) Fw(b) Fz(36px) Mb(-4px) D(ib)')
	quote_price_change = results.find('span', class_ = 'Trsdu(0.3s) Fw(500) Pstart(10px) Fz(24px) C($negativeColor)')
	quote_market = results.find('div', class_ = "C($tertiaryColor) D(b) Fz(12px) Fw(n) Mstart(0)--mobpsm Mt(6px)--mobpsm")

	if not quote_price_change:
		quote_price_change = results.find('span', class_ = 'Trsdu(0.3s) Fw(500) Pstart(10px) Fz(24px) C($positiveColor)')
	
	res_array.append(quote_name.text)
	res_array.append(quote_price.text)
	res_array.append(quote_price_change.text)
	res_array.append(quote_market.text)	

	print(res_array[0])
	print(res_array[1])
	print(res_array[2])
	print(res_array[3])
	return res_array
	
def get_stock_info(update, context):
	if context.args == []:
		context.bot.send_message(chat_id=update.effective_chat.id, text='''
			Please type in the ticker along with the /stock_info command
			Example: /stock_info AMZN''')
		return

	context.bot.send_message(chat_id = update.effective_chat.id, text = "Just a sec!")

	ticker = str(context.args[0])

	ticker_currency = check(ticker)

	URL = f'https://finance.yahoo.com/quote/{ticker_currency[1]}'

	q_arr = stock_info(URL)

	name_q = q_arr[0]
	price_q = q_arr[1]
	price_change_q = q_arr[2]
	market_q = q_arr[3]

	data = yf.download(tickers=ticker_currency[1], period='1y', interval='1d')
	close_data = data['Close']

	plt.figure(figsize=(20,10))
	plt.plot(close_data, label="Actual prices")
	plt.title(f'Prices of {name_q}', fontsize=20)
	#plt.xlabel('Time', fontsize=20)
	#plt.ylabel('Price', fontsize=20)
	plt.legend(loc='upper left', fontsize=20)
	plt.savefig('graph')

	msg = f'''
Current price of {ticker} is: 
{price_q}{ticker_currency[0]} {price_change_q}
{market_q}
'''

	context.bot.send_message(chat_id = update.effective_chat.id, text = msg)

	context.bot.send_photo(chat_id = update.effective_chat.id, photo = open('graph.png', 'rb'))
	
def abbreviation_list(update, context):
    context.bot.send_message(chat_id = update.effective_chat.id, text = '''
Here is the list:

Valuation Measures: 
PE (P/E) - Ratio of a company's share price to the company's earnings per share.
Trailing PE - Relative valuation multiple based on the last 12 months of actual earnings.
Forward PE - Version of the PE ratio that uses forecasted earnings for the PE calculation.
PEG (P/E to Growth ratio) - Ratio for comparing the price of a share to earnings per share and the company's expected future earnings.
EBITDA - A company's earnings before interest, taxes, depreciation, and amortization are subtracted.

Income statements:
EPS (Earnings per Share) - Monetary value of earnings per outstanding share of common stock for a company.
ROA (Return On Assets) - Percentage of how profitable a company's assets are in generating revenue.
ROE (Return On Equity) - Measure of the profitability of a business in relation to the equity.
PB (Price-To-Book) - Ratio used to compare a company's current market value to its book value.
(Book value is the value of all assets owned by a company)

Trading information:
Beta (β) - Measure of the volatility—or systematic risk—of a security or portfolio compared to the entire market.
Usually, the higher the beta, the greater the risk 
''')	

def stock_overview(ticker):
    url = "https://www.alphavantage.co/query?function=OVERVIEW&symbol=IBM&apikey=UONPPBM57DFDUNPV"

    querystring = {"function":"OVERVIEW","symbol":ticker}

    headers = {
        'x-rapidapi-key': "50dad0ebd4msh45efe476aefd6fcp185701jsn04e689c84052",
        'x-rapidapi-host': "alpha-vantage.p.rapidapi.com"
        }

    response = requests.request("GET", url, headers=headers, params=querystring)

    response_dict = json.loads(response.text)

    Industry = response_dict["Industry"]
    EBITDA = response_dict["EBITDA"]
    PE = response_dict["PERatio"]
    PEG = response_dict["PEGRatio"]
    EPS = response_dict["EPS"]
    ROA = response_dict["ReturnOnAssetsTTM"]
    ROE = response_dict["ReturnOnEquityTTM"]
    TrailPE = response_dict["TrailingPE"]
    ForwPe = response_dict["ForwardPE"]
    PB = response_dict["PriceToBookRatio"]
    Beta = response_dict["Beta"]
    Cap = response_dict["MarketCapitalization"]
    Currency = response_dict["Currency"]
    name = response_dict["Name"]

    msg = f'''
     Here is the general overview of {name}
    Industry: {Industry}

    Valuation Measures: 
    PE: {PE}
    Market capitalization: {Cap} {Currency}
    Trailing PE: {TrailPE}
    Forward PE: {ForwPe}
    PEG: {PEG}
    EBITDA: {EBITDA} {Currency}

    Income statements:
    EPS: {EPS}
    ROA: {ROA}
    ROE: {ROE}
    PB: {PB}

    Trading information:
    Beta: {Beta}

    To get the list of useful abbreviation type in:
    /abbreviation_list

    If you want to get fundamental analysis rating of {ticker} type in:
    /stock_analysis {ticker}
    '''

    return msg


def get_stock_overview(update, context):
	if context.args == []:
		context.bot.send_message(chat_id=update.effective_chat.id, text='Type in the quote along with the /stock_overview command')
		return

	context.bot.send_message(chat_id = update.effective_chat.id, text = "Just a sec!")

	prediction_text = stock_overview(str(context.args[0]))

	context.bot.send_message(chat_id = update.effective_chat.id, text = prediction_text)

def check(ticker):
	URL = f'https://finance.yahoo.com/quote/{ticker}'
	page = requests.get(URL)

	soup = BeautifulSoup(page.content, 'html.parser')

	results = soup.find(id='quote-header-info')

	tmp = []

	if not results:
		ticker = ticker + '.ME'
		tmp.append('₽')
		tmp.append(ticker)
		return tmp


	tmp.append('$')
	tmp.append(ticker)

	return tmp

def get_crypto_info(update, context):
	if context.args == []:
		context.bot.send_message(chat_id=update.effective_chat.id, text='Type in the quote along with the /crypto_info command')
		return

	context.bot.send_message(chat_id = update.effective_chat.id, text = "Just a sec!")
	ticker = str(context.args[0])

	URL = f'https://finance.yahoo.com/quote/{ticker}-USD?p={ticker}-USD'


	c_arr = stock_info(URL)
	name_c = c_arr[0]
	price_c = c_arr[1]
	price_change_c = c_arr[2]
	market_c = c_arr[3]

	msg = f'''
Current price of {ticker} is: 
{price_c}$ {price_change_c}
{market_c}
'''

	ticker_tmp = ticker + '-USD'

	data = yf.download(tickers=ticker+'-USD', period='6mo', interval='1d')
	close_data = data['Close']

	plt.figure(figsize=(20,10))
	plt.plot(close_data, label="Actual prices")
	plt.title(f'Prices of {ticker}', fontsize=20)
	#plt.xlabel('Time', fontsize=20)
	#plt.ylabel('Price', fontsize=20)
	plt.legend(loc='upper left', fontsize=20)
	plt.savefig('graph')

	context.bot.send_message(chat_id = update.effective_chat.id, text = msg)

	context.bot.send_photo(chat_id = update.effective_chat.id, photo = open('graph.png', 'rb'))
	

def data_type(update,context):
	if context.args == []:
		context.bot.send_message(chat_id=update.effective_chat.id, text='Type in the quote and model along with the /graphics command')
		return
	else:			
		global input_ticker_global
		input_ticker_global = str(context.args[0])
		global input_model_global
		input_model_global = str(context.args[1])


	list_of_data_type = ['15min (analysis)','60min (analysis)','1day (analysis)','1week (analysis)','1month (analysis)', '15min (prediction)','60min (prediction)']
	button_list = []
	for each in list_of_data_type:
		button_list.append(InlineKeyboardButton(each, callback_data = each))
	reply_markup=InlineKeyboardMarkup(build_menu(button_list,n_cols=3)) #n_cols = 1 is for single column and mutliple rows
	context.bot.send_message(chat_id=update.message.chat_id, text='Choose from the following intervals of data intake',reply_markup=reply_markup)


def build_menu(buttons,n_cols,header_buttons=None,footer_buttons=None):
	menu = [buttons[i:i + n_cols] for i in range(0, len(buttons), n_cols)]
	if header_buttons:
		menu.insert(0, header_buttons)
	if footer_buttons:
		menu.append(footer_buttons)
	return menu

def button_active(update,context) -> None:
	query = update.callback_query
	query.answer()

	ticker = input_ticker_global
	choice = query.data
	interval_choice = str(choice).split(' ')[0]
	timeline_choice = str(choice).split(' ')[1]
	df_active = create_df(ticker, interval_choice)

	if input_model_global == 'AR':
		funcAR(update,context,df_active,timeline_choice,interval_choice)
	if input_model_global == 'ARMA':
		funcARMA(update,context,df_active,timeline_choice,interval_choice)
	if input_model_global == 'ARIMA':
		funcARIMA(update,context,df_active,timeline_choice,interval_choice)
	if input_model_global == 'SARIMA':
		funcSARIMA(update,context,df_active,timeline_choice,interval_choice)	

def create_df(ticker, interval_choice):
	ts = TimeSeries('UONPPBM57DFDUNPV', output_format='csv')

	if interval_choice == '1day':
		Data=ts.get_daily(symbol = ticker, outputsize='full')
		df = pd.DataFrame(list(Data[0]))
	elif interval_choice == '1week':
		Data=ts.get_weekly(symbol = ticker)
		df = pd.DataFrame(list(Data[0]))
	elif interval_choice == '1month':
		Data=ts.get_monthly(symbol = ticker)
		df = pd.DataFrame(list(Data[0]))
	else:	
		totalData = ts.get_intraday(symbol = ticker, interval = str(interval_choice), outputsize = 'full')
		df = pd.DataFrame(totalData[0])

	df_1=df
	for i in range(1,len(df_1)):
		df_1.loc[i,1]= float(df_1[1][i])
		df_1.loc[i,0]=pd.to_datetime(df_1[0][i])
		df_1.loc[i,2]= float(df_1[2][i])
		df_1.loc[i,3]= float(df_1[3][i])
		df_1.loc[i,4]= float(df_1[4][i])
		df_1.loc[i,5]= float(df_1[5][i])
	input_df=df_1
	return input_df

def funcAR(update,context,df,timeline_choice,interval_choice):
	if timeline_choice == '(analysis)':
		ans_df = evaluate_models(df, [1, 2], [1], range(1, 3), 'AR')
		plt.figure(figsize=(25,15))
		plt.plot(df[1:int(len(df)*0.2)][0],ans_df, color="red", alpha=0.2)
	else:
		plt.figure(figsize=(25,15))
		plt.plot(extend_df(df,interval_choice)[2][::-1], evaluate_prediction_models(extend_df(df,interval_choice),"AR"), color="red", alpha=0.2)
	plt.title('AR Values', fontsize=20)
	plt.xlabel('Time', fontsize=14)
	plt.ylabel('Estimated Open Value', fontsize=14)	
	plt.savefig("figure_kek")
	context.bot.send_photo(chat_id = update.effective_chat.id, photo = open('figure_kek.png', 'rb'))

def funcARMA(update,context,df,timeline_choice,interval_choice):
	if timeline_choice == '(analysis)':
		ans_df = evaluate_models(df, [1, 2], [1], range(1, 3), 'ARMA')
		plt.figure(figsize=(25,15))
		plt.plot(df[1:int(len(df)*0.2)][0],ans_df, color="red", alpha=0.2)
	else:
		plt.figure(figsize=(25,15))
		plt.plot(extend_df(df,interval_choice)[2][::-1], evaluate_prediction_models(extend_df(df,interval_choice),"ARMA"), color="red", alpha=0.2)
	plt.title('ARMA Values', fontsize=20)
	plt.xlabel('Time', fontsize=14)
	plt.ylabel('Estimated Open Value', fontsize=14)		
	plt.savefig("figure_kek")
	context.bot.send_photo(chat_id = update.effective_chat.id, photo = open('figure_kek.png', 'rb'))

def funcARIMA(update,context,df,timeline_choice,interval_choice):
	if timeline_choice == '(analysis)':
		ans_df = evaluate_models(df, [1, 2], [1], range(1, 3), 'ARIMA')
		plt.figure(figsize=(25,15))
		plt.plot(df[1:int(len(df)*0.2)][0],ans_df, color="red", alpha=0.2)
	else:
		plt.figure(figsize=(25,15))
		plt.plot(extend_df(df,interval_choice)[2][::-1], evaluate_prediction_models(extend_df(df,interval_choice),"ARIMA"), color="red", alpha=0.2)
	plt.title('ARIMA Values', fontsize=20)
	plt.xlabel('Time', fontsize=14)
	plt.ylabel('Estimated Open Value', fontsize=14)		
	plt.savefig("figure_kek")
	context.bot.send_photo(chat_id = update.effective_chat.id, photo = open('figure_kek.png', 'rb'))

def funcSARIMA(update,context,df,timeline_choice,interval_choice):		
	if timeline_choice == '(analysis)':
		ans_df = evaluate_models(df, [1, 2], [1], range(1, 3), 'SARIMA')
		plt.figure(figsize=(25,15))
		plt.plot(df[1:int(len(df)*0.2)][0],ans_df, color="red", alpha=0.2)
	else:
		plt.figure(figsize=(25,15))
		plt.plot(extend_df(df,interval_choice)[2][::-1], formatter(evaluate_prediction_models(extend_df(df,interval_choice),"SARIMA")), color="red", alpha=0.2)
	plt.title('SARIMA Values', fontsize=20)
	plt.xlabel('Time', fontsize=14)
	plt.ylabel('Estimated Open Value', fontsize=14)		
	plt.savefig("figure_kek")
	context.bot.send_photo(chat_id = update.effective_chat.id, photo = open('figure_kek.png', 'rb'))


def analysis(ticker):
	API_extra = 'b97678e4b393a58564ba8d44c3ab979f'

	ratings = fa.rating(ticker, API_extra)

	rating_letter = str(ratings.iloc[0].at["rating"])
	rating_number = str(ratings.iloc[0].at["ratingScore"])
	rating_recommendation = str(ratings.iloc[0].at["ratingRecommendation"])

	rating_ROE = str(ratings.iloc[0].at["ratingDetailsROEScore"])
	rating_ROA = str(ratings.iloc[0].at["ratingDetailsROAScore"])	
	rating_PE = str(ratings.iloc[0].at["ratingDetailsPEScore"])
	rating_PB = str(ratings.iloc[0].at["ratingDetailsPBScore"])
	rating_DCF = str(ratings.iloc[0].at["ratingDetailsDCFScore"])
	rating_DE = str(ratings.iloc[0].at["ratingDetailsDEScore"])


	final = f'''
ROE rating: {rating_ROE}
ROA rating: {rating_ROA}
PE rating: {rating_PE}
PB rating: {rating_PB}
DE rating: {rating_DE}
DCF rating: {rating_DCF}

Current rating is: {rating_letter} ({rating_number})
The overall consensus is: {rating_recommendation}
	'''

	return final

def get_analysis(update, context):
	if context.args == []:
		context.bot.send_message(chat_id=update.effective_chat.id, text='Type in the quote along with the /stock_analysis command')
		return

	context.bot.send_message(chat_id = update.effective_chat.id, text = "Just a sec!")

	prediction_text = analysis(str(context.args[0]))

	context.bot.send_message(chat_id = update.effective_chat.id, text = prediction_text)


def exchange_rates(curr1, curr2):

	url = "https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=BTC&to_currency=CNY&apikey=UONPPBM57DFDUNPV"
	querystring = {"from_currency":curr1,"function":"CURRENCY_EXCHANGE_RATE","to_currency":curr2}

	headers = {
    	'x-rapidapi-key': "50dad0ebd4msh45efe476aefd6fcp185701jsn04e689c84052",
    	'x-rapidapi-host': "alpha-vantage.p.rapidapi.com"
    	}

	response = requests.request("GET", url, headers=headers, params=querystring)

	return(response)

def get_exchange_rates(update, context):
	if context.args == []:
		context.bot.send_message(chat_id=update.effective_chat.id, text='Type in the pair along with the /exchange command')
		return

	context.bot.send_message(chat_id = update.effective_chat.id, text = "Just a sec!")

	price_text = exchange_rates(str(context.args[0]), str(context.args[1]))
	
	response_dict = json.loads(price_text.text)
	response_info_dict = response_dict['Realtime Currency Exchange Rate']

	currFrom_code = response_info_dict['1. From_Currency Code']
	currFrom_name = response_info_dict['2. From_Currency Name']
	currTo_code = response_info_dict['3. To_Currency Code']
	currTo_name = response_info_dict['4. To_Currency Name']
	exchangeRate = response_info_dict['5. Exchange Rate']
	lastRefresh = response_info_dict['6. Last Refreshed']
	timeZone = response_info_dict['7. Time Zone']
	bidPrice = response_info_dict['8. Bid Price']
	askPrice = response_info_dict['9. Ask Price']


	exchange_msg = f'''
	Rate of {currFrom_code} to {currTo_code} : {exchangeRate}
	({currFrom_name} to {currTo_name})
	Bid: {bidPrice} Ask: {askPrice}
	Last refreshed: {lastRefresh} ({timeZone})'''


	context.bot.send_message(chat_id = update.effective_chat.id, text = exchange_msg)


def signal(update, context):

	context.bot.send_message(chat_id = update.effective_chat.id, text = "https://www.youtube.com/watch?v=dQw4w9WgXcQ")

def get_crypto_analysis(update, context):
    if context.args == []:
        context.bot.send_message(chat_id=update.effective_chat.id, text='''
    Please type in the ticker along with the /crypto_analysis command
    ''')
        return
    context.bot.send_message(chat_id = update.effective_chat.id, text = "Just a sec!")

    c_info = crypto_analysis(str(context.args[0]))

    context.bot.send_message(chat_id = update.effective_chat.id, text = c_info)

def crypto_analysis(ticker):
#url = "https://alpha-vantage.p.rapidapi.com/query"
    url = "https://www.alphavantage.co/query?function=CRYPTO_RATING&symbol=BTC&apikey=UONPPBM57DFDUNPV"

    #querystring = {"from_currency":"XRP","function":"CURRENCY_EXCHANGE_RATE","to_currency":"CAD"}
    querystring = {"symbol":ticker,"function":"CRYPTO_RATING"}

    headers = {
        'x-rapidapi-key': "50dad0ebd4msh45efe476aefd6fcp185701jsn04e689c84052",
           'x-rapidapi-host': "alpha-vantage.p.rapidapi.com"
        }

    response = requests.request("GET", url, headers=headers, params=querystring)

    response_dict = json.loads(response.text)

    if not 'Crypto Rating (FCAS)' in response_dict:
        return('Sorry, service is not avaliable right now :( ')

    response_info_dict = response_dict['Crypto Rating (FCAS)']


    name = response_info_dict['2. name']
    fcas_rating = response_info_dict['3. fcas rating']
    fcas_score = response_info_dict['4. fcas score']
    dev_score = response_info_dict['5. developer score']
    maturity_score = response_info_dict['6. market maturity score']
    utility_score = response_info_dict['7. utility score']

    msg = f'''
Today's ratings of {name} are:

FCAS rating: {fcas_score}
Developer Score: {dev_score}
Market Maturity Score: {maturity_score}
Utility Score: {utility_score}

Overall, the currency's health is {fcas_rating}'''

    return(msg)

def evaluate_ar(df):
    ans=[]
    df_train= df[int(len(df)*0.2):]
    df_test= df[1:int(len(df)*0.2)]
    predictions = []
    ar = tsa.AR(df_train[2].values)
    ar_fit = ar.fit()
    predictions = ar_fit.predict(len(df_train),  len(df_train)+len(df_test)-1, dynamic=False)
    ans.append(predictions)
    mae = mean_absolute_error(df_test[2], predictions)
    ans.append(mae)
    return ans

def evaluate_arima(df, model_order):
    ans=[]
    df_train= df[int(len(df)*0.2):]
    df_test= df[1:int(len(df)*0.2)]
    predictions = []
    arima = tsa.ARIMA(df_train[2].values, model_order)
    arima_fit = arima.fit(disp=0)
    predictions = arima_fit.predict(1, len(df_test), typ='levels')
    ans.append(predictions)
    mae = mean_absolute_error(df_test[2], predictions)
    ans.append(mae)
    return ans

def evaluate_arma(df, model_order):
    ans=[]
    df_train= df[int(len(df)*0.2):]
    df_test= df[1:int(len(df)*0.2)]
    predictions = []
    arma = tsa.ARMA(df_train[2].values, model_order)
    arma_fit = arma.fit(disp=0)
    predictions = arma_fit.predict(1, len(df_test), dynamic=False)
    ans.append(predictions)
    mae = mean_absolute_error(df_test[2], predictions)
    ans.append(mae)
    return ans

def evaluate_sarima(df, model_order):
    ans=[]
    df_train= df[int(len(df)*0.2):]
    df_test= df[1:int(len(df)*0.2)]
    predictions = []
    sarima=statespace.sarimax.SARIMAX(endog=(df_train[2].values.astype(float)),order=model_order,
    	seasonal_order=(model_order[0],model_order[1],model_order[2],12),trend='c',enforce_invertibility=False)
    sarima_fit = sarima.fit(disp=0)
    predictions = sarima_fit.predict(1, len(df_test), typ='levels')
    ans.append(predictions)
    mae = mean_absolute_error(df_test[2], predictions)
    ans.append(mae)
    return ans

def evaluate_models(df, p_values, d_values, q_values, choice):
    min_error=9999
    best_order=()
    best_prediction=[]
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                ans_list=[]
                index_dict = {}
                try:
                    if choice =='AR':
                        ans_list = evaluate_ar(df)
                    elif choice == 'ARIMA':
                        ans_list = evaluate_arima(df, order)
                    elif choice == 'ARMA':
                        order = (p,q)
                        ans_list = evaluate_arma(df, order)
                    elif choice == 'SARIMA':
                        ans_list = evaluate_sarima(df,order)
                    elif choice == 'Best_Model':
                        ans_list_arima = evaluate_arima(df, order)
                        ans_list_arima.append('ARIMA')
                        index_dict[ans_list_arima[1]] = ans_list_arima
                        
                        ans_list_arma = evaluate_arma(df, (p,q))
                        ans_list_arma.append('ARMA')
                        index_dict[ans_list_arma[1]] = ans_list_arma
                        
                        ans_list_sarima = evaluate_sarima(df,order)
                        ans_list_sarima.append('SARIMA')
                        index_dict[ans_list_sarima[1]] = ans_list_sarima
                        
                        ans_list_ar = evaluate_ar(df)
                        ans_list_ar.append('AR')
                        index_dict[ans_list_ar[1]] = ans_list_ar
                        
                        ans_list.append(index_dict[min(index_dict.keys())][0])
                        ans_list.append([min(index_dict.keys())])
                        ans_list.append(index_dict[min(index_dict.keys())][2])      
                    prediction = ans_list[0]
                    error = ans_list[1]
                    if error < min_error:
                        min_error, best_order,best_prediction = error, order, prediction
                    #print('ARIMA%s MAE=%.3f' % (order,error))
                except:
                    continue
    #print('Best ARIMA%s MAE=%.3f' % (best_order, min_error))
    if choice != 'Best_Model':
        return best_prediction
    elif choice == 'Best_Model':
        return ans_list

def extend_df(df,frequency):
    ans=[]
    df_1=df
    valuey=0
    valuem=1
    valued=0
    valueh=0
    valuemin=0
    values=0
    endo=pd.Timestamp(df_1[0][1].year+valuey,df_1[0][1].month+valuem,df_1[0][1].day+valued,df_1[0][1].hour+valueh,df_1[0][1].minute+valuemin,df_1[0][1].second+values)
    df_2 = pd.date_range(start=df_1[0][1] ,end=endo, freq=pd.offsets.BusinessHour(start="4:00",end="21:00")).to_series().to_frame(0)
    df_7 = pd.DataFrame(np.array([0]),columns=[0])
    df_2=df_2[1:]
    if frequency=='15min':
        for i in range(0,len(df_2),17):
            df_3 = pd.date_range(start=df_2[0][i],end=df_2[0][i+16], freq='15min').to_series().to_frame(0)
            df_7 = pd.concat([df_7, df_3], ignore_index=True)
        df_7=df_7[1:]
        df_6=df_1[1:]
        df_3=pd.concat([df_7[::-1], df_6], ignore_index=True)
    if frequency=='60min':
        df_7=df_2
        df_6=df_1[1:]
        df_3=pd.concat([df_7[::-1], df_6], ignore_index=True)
    ans.append(df_3)
    ans.append(df[1:])
    ans.append(df_7)
    return ans

def evaluate_pred_arima(df):
    df_train=df[1]
    df_test= df[2][::-1]
    arima = tsa.ARIMA(df_train[2].values, (2,1,2))
    arima_fit = arima.fit(disp=0)
    predictions = arima_fit.predict(1, len(df_test), typ='levels')
    return predictions

def formatter(df):
    for i in range(len(df)):
        if df[i]>np.mean(df)*(7/6):
            df[i]=np.mean(df)
    return df

def evaluate_pred_ar(df):
    df_train= df[1]
    df_test= df[2][::-1]
    ar = tsa.AR(df_train[2].values)
    ar_fit = ar.fit()
    predictions = ar_fit.predict(len(df_train),  len(df_train)+len(df_test)-1, dynamic=False)
    return predictions

def evaluate_pred_arma(df):
    df_train= df[1]
    df_test= df[2][::-1]
    arma = tsa.ARMA(df_train[2].values, (2,2))
    arma_fit = arma.fit(disp=0)
    predictions = arma_fit.predict(1, len(df_test), dynamic=False)
    return predictions

def evaluate_pred_sarima(df):
    df_train= df[1]
    df_test= df[2][::-1]
    sarima=statespace.sarimax.SARIMAX(endog=(df_train[2].values.astype(float)),order=(2,1,2),seasonal_order=(2,1,2,12),trend='c',enforce_invertibility=False)
    sarima_fit = sarima.fit(disp=0)
    predictions = sarima_fit.predict(1, len(df_test), typ='levels')
    return predictions

def evaluate_prediction_models(df,choice):
    if choice =='AR':
        prediction = evaluate_pred_ar(df)
    elif choice == 'ARIMA':
        prediction = evaluate_pred_arima(df)
    elif choice == 'ARMA':
        prediction = evaluate_pred_arma(df)
    elif choice == 'SARIMA':
        prediction = evaluate_pred_sarima(df)
    return prediction   


def main():
	print('bot pashet')
	TOKEN = os.getenv("1873581201:AAHmTEOSmaiRuxsOTsuw28oItvI09Tt5rxI")
	updater = Updater("1873581201:AAHmTEOSmaiRuxsOTsuw28oItvI09Tt5rxI", use_context=True)
    
	dp = updater.dispatcher

	start_handler = CommandHandler('start', start)
	dp.add_handler(start_handler)

	help_handler = CommandHandler('help', help)
	dp.add_handler(help_handler)

	quote_handler = CommandHandler('stock_info', get_stock_info)
	dp.add_handler(quote_handler)

	crypto_handler = CommandHandler('crypto_info', get_crypto_info)
	dp.add_handler(crypto_handler)

	buttons_handler = CommandHandler('graphics', data_type)
	dp.add_handler(buttons_handler)

	prediction_handler = CommandHandler('stock_analysis', get_analysis)
	dp.add_handler(prediction_handler)

	overview_handler = CommandHandler('stock_overview', get_stock_overview)
	dp.add_handler(overview_handler)

	exchange_handler = CommandHandler('exchange', get_exchange_rates)
	dp.add_handler(exchange_handler)

	abbreviation_handler = CommandHandler('abbreviation_list', abbreviation_list)
	dp.add_handler(abbreviation_handler)

	crypto_analysis_handler = CommandHandler('crypto_analysis', get_crypto_analysis)
	dp.add_handler(crypto_analysis_handler)

	signal_handler = CommandHandler('get_signals', signal)
	dp.add_handler(signal_handler)

	button_callback_handler = CallbackQueryHandler(button_active)
	dp.add_handler(button_callback_handler)


	updater.start_polling()
	updater.idle()
    

if __name__ == '__main__':
    main()