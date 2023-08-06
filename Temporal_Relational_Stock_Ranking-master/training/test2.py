import json

def calculate_average(results):
    avg_results = {}
    num_sets = len(results)

    for idx, result_str in enumerate(results):
        result_list = json.loads(result_str)  # Convert the string to a list of dictionaries
        num_runs = len(result_list)

        avg_mse = sum(result[0] for result in result_list) / num_runs
        avg_mrrt = sum(result[1] for result in result_list) / num_runs
        avg_btl = sum(result[2] for result in result_list) / num_runs

        avg_results[f'set_{idx + 1}_mse'] = avg_mse
        avg_results[f'set_{idx + 1}_mrrt'] = avg_mrrt
        avg_results[f'set_{idx + 1}_btl'] = avg_btl

    return avg_results

result_str1 = '{"mse": 0.00037784899098915913, "mrrt": 0.027345894375467396, "btl": 0.9153886415879242}'
result_str2 = '{"mse": 0.00029991223467873184, "mrrt": 0.028201345652124187, "btl": 0.9123456789012345}'

result_strings = [result_str1, result_str2]
print(calculate_average(result_strings))
