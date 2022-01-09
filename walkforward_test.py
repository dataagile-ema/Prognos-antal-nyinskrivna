from forecast_test import ErrorMetrics, create_ForecastTest, run_ForecastTest
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def create_walkforward_forecast_test(start_date, df, no_of_days_in_forecast, n, label_for_predicted_variable):
    start_date_list = [start_date + pd.Timedelta(days=i*no_of_days_in_forecast) for i in range(n)]

    start_date_plot = start_date - pd.Timedelta(days=no_of_days_in_forecast)
    end_date_plot = start_date + pd.Timedelta(days=n*no_of_days_in_forecast)

    tests_list = [create_ForecastTest(df, 
                        start_date_list[i], 
                        no_of_days_in_forecast, 
                        label_for_predicted_variable, 
                        start_date_plot,
                        end_date_plot) for i in range(n)]

    return tests_list

def run_walkforward_ForecastTest(tests_list):
    """"
    Run the tests for the walkforward_forecast_test
    tests_list: list of ForecastTest objects
    Returns: list of ErrorMetrics objects
    """
    results_model = [run_ForecastTest(tests_list[i], False) for i in range(len(tests_list))]
    results_naive = [run_ForecastTest(tests_list[i], True) for i in range(len(tests_list))]

    return results_model, results_naive


def calc_mean_MEA(results_model, results_naive):
    mean_MEA_model = np.mean([results_model[i].MAE for i in range(len(results_model))])
    mean_MEA_naive = np.mean([results_naive[i].MAE for i in range(len(results_naive))])
    return mean_MEA_model, mean_MEA_naive


def calc_mean_RMSE(results_model, results_naive):
    mean_RMSE_model = np.mean([results_model[i].RMSE for i in range(len(results_model))])
    mean_RMSE_naive = np.mean([results_naive[i].RMSE for i in range(len(results_naive))])
    return mean_RMSE_model, mean_RMSE_naive

def calc_mean_EB(results_model, results_naive):
    mean_EB_model = np.mean([results_model[i].EB for i in range(len(results_model))])
    mean_EB_naive = np.mean([results_naive[i].EB for i in range(len(results_naive))])
    return mean_EB_model, mean_EB_naive


def plot_results_walkforward(results_model, results_naive):
    mean_MEA_model, mean_MEA_naive = calc_mean_MEA(results_model, results_naive)
    mean_RMSE_model, mean_RMSE_naive = calc_mean_RMSE(results_model, results_naive)
    mean_EB_model, mean_EB_naive = calc_mean_EB(results_model, results_naive)

    # get max value of all mean values
    max_value_y_scale = max(mean_MEA_model, mean_MEA_naive, mean_RMSE_model, mean_RMSE_naive, mean_EB_model, mean_EB_naive) + 1
    # get min value of all mean values
    min_value_y_scale = min(mean_MEA_model, mean_MEA_naive, mean_RMSE_model, mean_RMSE_naive, mean_EB_model, mean_EB_naive) - 1

    # Medel absolut fel (MAE)
    print('Mean MAE for model: {:.1f}'.format(mean_MEA_model))
    print('Mean MAE for naive: {:.1f}'.format(mean_MEA_naive))

    plt.bar(['Model', 'Naiv'], [mean_MEA_model, mean_MEA_naive])
    plt.ylim(min_value_y_scale, max_value_y_scale)
    plt.title('Medel absolut fel')
    plt.show()   
   
    # Medel RMSE
    print('Medel RMSE modell: {:.1f}'.format(mean_RMSE_model))
    print('Medel RMSE naive: {:.1f}'.format(mean_RMSE_naive))

    plt.bar(['Model', 'Naiv'], [mean_RMSE_model, mean_RMSE_naive])
    plt.ylim(min_value_y_scale, max_value_y_scale)
    plt.title('Medel RMSE')
    plt.show()

    # Medelvärde för alla residualer (EB)
    print('Medelvärde alla residualer modell: {:.1f}'.format(mean_EB_model))
    print('Medelvärde alla residualer  naive: {:.1f}'.format(mean_EB_naive))

    plt.bar(['Model', 'Naiv'], [mean_EB_model, mean_EB_naive])
    plt.ylim(min_value_y_scale, max_value_y_scale)    
    plt.title('Medelvärde alla residualer')
    plt.show()

    






