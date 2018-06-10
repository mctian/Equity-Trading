import sys
sys.path.append('/Users/masontian/Documents/GitHub/Equity-Analysis')
import Model
import numpy as np


from zipline.api import order_target_percent, record, symbol, date_rules, time_rules
from zipline.finance import commission, slippage, execution


def initialize(context):
	context.i = 99

	# Explicitly set the commission/slippage to the "old" value until we can
	# rebuild example data.
	# github.com/quantopian/zipline/blob/master/tests/resources/
	# rebuild_example_data#L105
	context.set_commission(commission.PerShare(cost=.005, min_trade_cost=1.0))
	context.set_slippage(slippage.VolumeShareSlippage())
	context.schedule_function(func = run, date_rule = date_rules.month_start(), time_rule = time_rules.market_open(), half_days = True, calendar = None)


# make a portfolio with predicted probabilities higher than a hardcoded threshold
def makePortfolio(index, randForest):
	featureList = ['EPS Growth', 'Volatility 180 D', 'Trailing EPS', 'Price to Cash Flow', 'EPS', 'Volume', 'Return on Assets', 'Price to Book', 'Dividend Yield', 'Total Debt to Total Equity', 'Return on Invested Capital', 'Return on Common Equity']
	addedStocks, probabilities = Model.predict_probabilities(randForest, startIndex = -1 * index - 11, endIndex = -1 * index, features = featureList, sector = "Health Care")
	probabilityThreshold = 0.8
	stockTuples = zip(addedStocks, probabilities)
	stockTuples = list(filter(lambda x: x[1][1] > probabilityThreshold, stockTuples))
	if len(stockTuples) == 0:
		print("No portfolio, probabilities lower than threshold of " + str(probabilityThreshold))
		return 0
	stocks, probabilities = zip(*stockTuples)
	return stocks


def handle_data(context, data):
	return


def run(context, data):
	if context.i % 3 == 0:
		indexes = np.arange(-15 - context.i + 200 * -1, -15 - context.i, 3)
		sector = "Health Care"
		featureList = ['EPS Growth', 'Volatility 180 D', 'Trailing EPS', 'Price to Cash Flow', 'EPS', 'Volume', 'Return on Assets', 'Price to Book', 'Dividend Yield', 'Total Debt to Total Equity', 'Return on Invested Capital', 'Return on Common Equity']
		randForest = Model.buildWithIndexesTripleClass(modelType = Model.randomForestClassifier, indexes = indexes, target= 'Rate of Return', features = featureList, featureLength = 12,\
									targetLength = 3, sector = sector, percentileTarget = 90, percentileAvoid = 10, verbose = True)
		portfolio = makePortfolio(context.i, randForest)
		for stock in portfolio:
			order_target_percent(asset = symbol(stock), target = 1/len(portfolio), style = execution.MarketOrder())
	context.i -= 1
	return
