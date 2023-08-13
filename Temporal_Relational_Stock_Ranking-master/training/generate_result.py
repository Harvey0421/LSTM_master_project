import csv
from rank_lstm import RankLSTM
from relation_rank_lstm import ReRaLSTM
from relation_rank_price_lstm import ReRaPrLSTM

def run_training_with_hyperparameters(hyperparameters):
    # Replace this function with your actual training code.
    # It should take the hyperparameters as input and return the results.
    # For this example, I'll just return some random results.
    import random
    return {
        'mse': random.uniform(0.0001, 0.001),
        'mrrt': random.uniform(0.01, 0.1),
        'btl': random.uniform(0.8, 0.95)
    }

def save_results_to_csv(results, avg_results, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Hyperparameter Set', 'Run', 'MSE', 'MRRt', 'BTL'])
        for idx, (hyperparameters, result_set) in enumerate(results.items()):
            for run_num, result in enumerate(result_set):
                row_data = [f'Set {idx + 1}', run_num + 1, result['mse'], result['mrrt'], result['btl']]
                writer.writerow(row_data)

            # Write the average for each hyperparameter set
            avg_row_data = ['Average', '', avg_results[f'set_{idx+1}_mse'], avg_results[f'set_{idx+1}_mrrt'], avg_results[f'set_{idx+1}_btl']]
            writer.writerow(avg_row_data)
            writer.writerow([])  # Add an empty row to separate results from the next hyperparameter set

def calculate_average(results):
    avg_results = {}
    num_sets = len(results)

    for idx, result_set in enumerate(results.values()):
        num_runs = len(result_set)
        avg_mse = sum(result['mse'] for result in result_set) / num_runs
        avg_mrrt = sum(result['mrrt'] for result in result_set) / num_runs
        avg_btl = sum(result['btl'] for result in result_set) / num_runs

        avg_results[f'set_{idx + 1}_mse'] = avg_mse
        avg_results[f'set_{idx + 1}_mrrt'] = avg_mrrt
        avg_results[f'set_{idx + 1}_btl'] = avg_btl

    return avg_results

def main():
    # Define the hyperparameter sets here as a list of dictionaries
    market=['NASDAQ','NYSE']
    tick=['NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv','NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv']
    rel =['wikidata', 'sector_industry']
    parameter = {'seq': 4, 'unit': 64, 'lr': 0.001, 'alpha': 1}
    hyperparameter_sets = [
        #RankLSTM
        #{'data_path': '../data/2013-01-01', 'market_name': market[0], 'tickers_fname': tick[0], 'parameters': parameter},
        #{'data_path': '../data/2013-01-01', 'market_name': market[1], 'tickers_fname': tick[1], 'parameters': parameter},
        #ReRaLSTM
        #{'data_path': '../data/2013-01-01', 'market_name': market[0], 'tickers_fname': tick[0], 'relation':rel[0], 'parameters': parameter},
        #{'data_path': '../data/2013-01-01', 'market_name': market[1], 'tickers_fname': tick[1], 'relation': rel[0], 'parameters': parameter},
        #{'data_path': '../data/2013-01-01', 'market_name': market[0], 'tickers_fname': tick[0], 'relation': rel[1], 'parameters': parameter},
        #{'data_path': '../data/2013-01-01', 'market_name': market[1], 'tickers_fname': tick[1], 'relation': rel[1], 'parameters': parameter},
        #ReRaPrLSTM
        {'data_path': '../data/2013-01-01', 'market_name': market[0], 'tickers_fname': tick[0], 'relation': rel[0], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[0] + '_correlation_graph_0.9.json'},
        {'data_path': '../data/2013-01-01', 'market_name': market[0], 'tickers_fname': tick[0], 'relation': rel[0], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[0] + '_correlation_graph_0.925.json'},
        {'data_path': '../data/2013-01-01', 'market_name': market[0], 'tickers_fname': tick[0], 'relation': rel[0], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[0] + '_correlation_graph_0.95.json'},
        {'data_path': '../data/2013-01-01', 'market_name': market[0], 'tickers_fname': tick[0], 'relation': rel[0], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[0] + '_correlation_graph_0.975.json'},
        {'data_path': '../data/2013-01-01', 'market_name': market[0], 'tickers_fname': tick[0], 'relation': rel[0], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[0] + '_correlation_graph_0.99.json'},
        {'data_path': '../data/2013-01-01', 'market_name': market[0], 'tickers_fname': tick[0], 'relation': rel[1], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[0] + '_correlation_graph_0.9.json'},
        {'data_path': '../data/2013-01-01', 'market_name': market[0], 'tickers_fname': tick[0], 'relation': rel[1], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[0] + '_correlation_graph_0.925.json'},
        {'data_path': '../data/2013-01-01', 'market_name': market[0], 'tickers_fname': tick[0], 'relation': rel[1], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[0] + '_correlation_graph_0.95.json'},
        {'data_path': '../data/2013-01-01', 'market_name': market[0], 'tickers_fname': tick[0], 'relation': rel[1], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[0] + '_correlation_graph_0.975.json'},
        {'data_path': '../data/2013-01-01', 'market_name': market[0], 'tickers_fname': tick[0], 'relation': rel[1], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[0] + '_correlation_graph_0.99.json'},
        {'data_path': '../data/2013-01-01', 'market_name': market[1], 'tickers_fname': tick[1], 'relation': rel[0], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[1] + '_correlation_graph_0.9.json'},
        {'data_path': '../data/2013-01-01', 'market_name': market[1], 'tickers_fname': tick[1], 'relation': rel[0], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[1] + '_correlation_graph_0.925.json'},
        {'data_path': '../data/2013-01-01', 'market_name': market[1], 'tickers_fname': tick[1], 'relation': rel[0], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[1] + '_correlation_graph_0.95.json'},
        {'data_path': '../data/2013-01-01', 'market_name': market[1], 'tickers_fname': tick[1], 'relation': rel[0], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[1] + '_correlation_graph_0.975.json'},
        {'data_path': '../data/2013-01-01', 'market_name': market[1], 'tickers_fname': tick[1], 'relation': rel[0], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[1] + '_correlation_graph_0.99.json'},
        {'data_path': '../data/2013-01-01', 'market_name': market[1], 'tickers_fname': tick[1], 'relation': rel[1], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[1] + '_correlation_graph_0.9.json'},
        {'data_path': '../data/2013-01-01', 'market_name': market[1], 'tickers_fname': tick[1], 'relation': rel[1], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[1] + '_correlation_graph_0.925.json'},
        {'data_path': '../data/2013-01-01', 'market_name': market[1], 'tickers_fname': tick[1], 'relation': rel[1], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[1] + '_correlation_graph_0.95.json'},
        {'data_path': '../data/2013-01-01', 'market_name': market[1], 'tickers_fname': tick[1], 'relation': rel[1], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[1] + '_correlation_graph_0.975.json'},
        {'data_path': '../data/2013-01-01', 'market_name': market[1], 'tickers_fname': tick[1], 'relation': rel[1], 'parameters': parameter, 'new_relation_graph': '../data/price_graph/' + market[1] + '_correlation_graph_0.99.json'},
    ]
    for i in range(0, len(hyperparameter_sets)):
        print(hyperparameter_sets[i])

    results = {}
    total=len(hyperparameter_sets)*5
    current=1
    for idx, hyperparameters in enumerate(hyperparameter_sets):
        result_set = []
        if idx < 0:
            for run_num in range(5):
                print(f'Running set {idx + 1} of {len(hyperparameter_sets)} ({current} of {total})')
                current+=1
                rank_LSTM = RankLSTM(data_path=hyperparameters['data_path'],
                                    market_name=hyperparameters['market_name'],
                                    tickers_fname=hyperparameters['tickers_fname'],
                                    parameters=hyperparameters['parameters'],
                                    steps=1, epochs=50, batch_size=None, gpu=1)
                result = rank_LSTM.train()
                result_set.append(result[-1])
            results[f'Set {idx + 1}'] = result_set
        elif idx < 0 and idx >= 0:
            for run_num in range(5):
                print(f'Running set {idx + 1} of {len(hyperparameter_sets)} ({current} of {total})')
                current+=1
                RR_LSTM = ReRaLSTM(
                    data_path=hyperparameters['data_path'],
                    market_name=hyperparameters['market_name'],
                    tickers_fname=hyperparameters['tickers_fname'],
                    relation_name=hyperparameters['relation'],
                    parameters=hyperparameters['parameters'],
                    steps=1, epochs=50, batch_size=None, gpu=1, in_pro=False
                )
                result = RR_LSTM.train()
                result_set.append(result[-1])
            results[f'Set {idx + 1}'] = result_set
        else:
            for run_num in range(5):
                print(f'Running set {idx + 1} of {len(hyperparameter_sets)} ({current} of {total})')
                current+=1
                RRP_LSTM = ReRaPrLSTM(
                    data_path=hyperparameters['data_path'],
                    market_name=hyperparameters['market_name'],
                    tickers_fname=hyperparameters['tickers_fname'],
                    relation_name=hyperparameters['relation'],
                    parameters=hyperparameters['parameters'],
                    steps=1, epochs=50, batch_size=None, gpu=1,
                    in_pro=False,
                    new_relation_graph=hyperparameters['new_relation_graph']
                )
                result = RRP_LSTM.train()
                result_set.append(result[-1])
            results[f'Set {idx + 1}'] = result_set
    print(results)
    avg_results = calculate_average(results)
    save_results_to_csv(results, avg_results, 'training_results.csv')


if __name__ == '__main__':
    main()
