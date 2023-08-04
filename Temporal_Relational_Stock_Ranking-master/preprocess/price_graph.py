import os
import numpy as np
from datetime import datetime
import json
from scipy.stats import pearsonr

class EOD_Preprocessor:
    def __init__(self, data_path, market_name):
        self.data_path = data_path
        self.date_format = '%Y-%m-%d %H:%M:%S'
        self.market_name = market_name

    def _read_EOD_data(self):
        self.data_EOD = []
        for index, ticker in enumerate(self.tickers):
            single_EOD = np.genfromtxt(
                os.path.join(self.data_path, self.market_name + '_' + ticker +
                             '_30Y.csv'), dtype=str, delimiter=',',
                skip_header=True
            )
            self.data_EOD.append(single_EOD)
        print('#stocks\' EOD data readin:', len(self.data_EOD))
        assert len(self.tickers) == len(self.data_EOD), 'length of tickers ' \
                                                        'and stocks not match'

    def _read_tickers(self, ticker_fname):
        self.tickers = np.genfromtxt(ticker_fname, dtype=str, delimiter='\t',
                                     skip_header=True)[:, 0]

    def _transfer_EOD_str(self, selected_EOD_str, tra_date_index):
        selected_EOD = np.zeros(selected_EOD_str.shape, dtype=float)
        for row, daily_EOD in enumerate(selected_EOD_str):
            date_str = daily_EOD[0].replace('-05:00', '')
            date_str = date_str.replace('-04:00', '')
            selected_EOD[row][0] = tra_date_index[date_str]
            for col in range(1, selected_EOD_str.shape[1]):
                selected_EOD[row][col] = float(daily_EOD[col])
        return selected_EOD

    def _calculate_price_similarity_matrix(self, closing_prices):
        # Transpose closing_prices to have each row representing a stock and each column representing a time step
        transposed_prices = closing_prices.T

        # Calculate the correlation matrix between the closing prices of all stocks
        correlation_matrix = np.corrcoef(transposed_prices)

        # The similarity matrix is just the correlation matrix
        return correlation_matrix

    def _create_graph_from_similarity(self, similarity_matrix, correlation_threshold):
        graph = {}
        for i in range(similarity_matrix.shape[0]):
            graph[self.tickers[i]] = []
            for j in range(similarity_matrix.shape[1]):
                if i != j and similarity_matrix[i, j] >= correlation_threshold:
                    graph[self.tickers[i]].append(self.tickers[j])
        return graph

    def generate_correlation_graph(self, selected_tickers_fname, begin_date, correlation_threshold=0.9, output_file=None):
        trading_dates = np.genfromtxt(
            os.path.join(self.data_path, '..',
                         self.market_name + '_aver_line_dates.csv'),
            dtype=str, delimiter=',', skip_header=False
        )
        print('#trading dates:', len(trading_dates))
        # begin_date = datetime.strptime(trading_dates[29], self.date_format)
        print('begin date:', begin_date)
        # transform the trading dates into a dictionary with index
        index_tra_dates = {}
        tra_dates_index = {}
        for index, date in enumerate(trading_dates):
            tra_dates_index[date] = index
            index_tra_dates[index] = date
        self.tickers = np.genfromtxt(
            os.path.join(self.data_path, '..', selected_tickers_fname),
            dtype=str, delimiter='\t', skip_header=False
        )
        print('#tickers selected:', len(self.tickers))
        self._read_EOD_data()

        closing_prices = np.zeros((len(trading_dates), len(self.tickers)))
        #print('data_EOD:',self.data_EOD)

        for stock_index, single_EOD in enumerate(self.data_EOD):
            # Select data within the begin_date
            begin_date_row = -1
            for date_index, daily_EOD in enumerate(single_EOD):
                date_str = daily_EOD[0].replace('-05:00', '')
                date_str = date_str.replace('-04:00', '')
                cur_date = datetime.strptime(date_str, self.date_format)
                if cur_date > begin_date:
                    begin_date_row = date_index
                    break

            selected_EOD_str = single_EOD[begin_date_row:]
            selected_EOD = self._transfer_EOD_str(selected_EOD_str, tra_dates_index)

            # Extract the closing prices for the current stock and store them in the corresponding column
            closing_prices[:len(selected_EOD), stock_index] = selected_EOD[:, 4]


        # Step 1: Calculate the similarity matrix based on price movements (Pearson correlation)
        similarity_matrix = self._calculate_price_similarity_matrix(closing_prices)

        # Step 2: Create the graph based on the correlation threshold
        graph = self._create_graph_from_similarity(similarity_matrix, correlation_threshold)

        if output_file:
            with open(output_file, 'w') as graph_file:
                json.dump(graph, graph_file)
        return graph


if __name__ == '__main__':
    data_path = '../data/google_finance'
    market_name = 'NYSE'
    selected_tickers_fname = market_name+'_tickers_qualify_dr-0.98_min-5_smooth.csv'
    begin_date = datetime.strptime('2012-11-19 00:00:00', '%Y-%m-%d %H:%M:%S')
    correlation_threshold = 0.99
    output_file = '../data/price_graph/'+market_name+'_correlation_graph_'+str(correlation_threshold)+'.json'

    processor = EOD_Preprocessor(data_path, market_name)
    correlation_graph = processor.generate_correlation_graph(selected_tickers_fname, begin_date, correlation_threshold, output_file)

    # Print the correlation graph
    print(correlation_graph)