from test_forecast import ErrorMetrics, create_ForecastTest, run_ForecastTest
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
def create_walkforward_forecast_test(start_date, df, no_of_days_in_forecast, n, label_for_predicted_variable):
    start_date_list = [start_date + pd.Timedelta(days=i*no_of_days_in_forecast) for i in range(n)]

    tests_list = [create_ForecastTest(df, start_date_list[i], no_of_days_in_forecast, label_for_predicted_variable) for i in range(n)]
    return tests_list

# define function that run_ForecastTest on a list of ForecastTest-objects
def run_walkforward_ForecastTest(tests_list):
    results_model = [run_ForecastTest(tests_list[i], False) for i in range(len(tests_list))]
    results_naive = [run_ForecastTest(tests_list[i], True) for i in range(len(tests_list))]

    return results_model, results_naive

# define function that calculates the mean MEA for results_model and results_naive
def calc_mean_MEA(results_model: ErrorMetrics, results_naive: ErrorMetrics):
    mean_MEA_model = np.mean([results_model[i].MAE for i in range(len(results_model))])
    mean_MEA_naive = np.mean([results_naive[i].MAE for i in range(len(results_naive))])
    return mean_MEA_model, mean_MEA_naive

# define function that calculates the mean RMSE for results_model and results_naive
def calc_mean_RMSE(results_model: ErrorMetrics, results_naive: ErrorMetrics):
    mean_RMSE_model = np.mean([results_model[i].RMSE for i in range(len(results_model))])
    mean_RMSE_naive = np.mean([results_naive[i].RMSE for i in range(len(results_naive))])
    return mean_RMSE_model, mean_RMSE_naive

def plot_results_walkforward(results_model: ErrorMetrics, results_naive: ErrorMetrics):
    # print mean of MEA for model and naive with discriptions
    mean_MEA_model, mean_MEA_naive = calc_mean_MEA(results_model, results_naive)
    # print mean of mean_MEA_model and mean_MEA_naive with 1 decimal
    print('Mean MAE for model: {:.1f}'.format(mean_MEA_model))
    print('Mean MAE for naive: {:.1f}'.format(mean_MEA_naive))

    # barplot mean_MEA_model and mean_MEA_naive
    plt.bar(['Model', 'Naiv'], [mean_MEA_model, mean_MEA_naive])
    plt.title('Medel absolut fel')
    plt.show()   
   
  
    # print mean of RMSE for model and naive with discriptions
    mean_RMSE_model, mean_RMSE_naive = calc_mean_RMSE(results_model, results_naive)
    print('Medel RMSE fel modell: {:.1f}'.format(mean_RMSE_model))
    print('Medel RMSE fel naive: {:.1f}'.format(mean_RMSE_naive))


    # barplot mean_RMSE_model and mean_RMSE_naive
    plt.bar(['Model', 'Naiv'], [mean_RMSE_model, mean_RMSE_naive])
    plt.title('Medel relativt fel')
    plt.show()








