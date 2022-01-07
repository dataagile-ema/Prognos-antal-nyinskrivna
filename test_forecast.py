import pandas as pd
import fbprophet as fb
import matplotlib.pyplot as plt
from dataclasses import dataclass



@dataclass
class ForecastTest:
    """"
    Parametrar för att göra ett test av prognos av en tidsserie.
    start_of_prediction: datum då prognosen ska börja
    end_of_prediction: datum då prognosen ska slutar
    period: antal dagar som ska prognoseras
    df_train: dataframe med data som modellen passas med
    df: hela datasetet som läses in från csv-filen, som testet använder som träningsdata och utfall.
    """
    start_of_forecast: pd.Timestamp 
    end_of_forecast: pd.Timestamp
    period: pd.Timedelta
    df_train: pd.DataFrame
    df: pd.DataFrame
    label_for_predicted_variable: str

# dateclass error metrics
@dataclass 
class ErrorMetrics:
    """
    mae: mean absolute error
    rmse: root mean square error
    eb: error bias
    """
    MAE: float
    RMSE: float
    EB: float

@dataclass 
class ForecastResults:
    """
    Parametrar för att visa resultat av prognos.
    df_forecast: dataframe med prognosen
    df_naive_forecast: dataframe med utfall från naive modellen (medelvärdet av antal nyinskrivna för sista perioden in träningsdata)
    model: modell-objektet
    df_residuals: dataframe med residualer
    MAE: float medel av de absoluta värdena på residualerna
    RMSE: float roten av medel av de kvadraterna på residualerna
    EB: float bias för felet mellan prognos och utfall
    """
    df_forecast: pd.DataFrame
    model: fb.Prophet
    df_residuals: pd.DataFrame
    errorMetrics: ErrorMetrics



def read_and_prep_data(path: str):
    """
    läser csv fil med format ds (datum) och y (antal nyinskrivna)
    """
    df = pd.read_csv(path)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values(by='ds')
    return df


from datetime import timedelta, date

def create_ForecastTest(df: pd.DataFrame, forecast_period_start_date: pd.Timestamp, 
        no_of_days_in_forecast: int, label_for_predicted_variable: str) -> ForecastTest:
    """
    Skapar ett objekt med paramterar för ett testa prognos
    df: hela datasetet som läses in från csb-filen, som testet använder som träningsdata och utfall.
    pred_period_start_date: datum då prognosen ska börja
    prediction_period: antal dagar som ska prognoseras (exemple: 15)
    """
    test = ForecastTest(None, None, None, None, None, None)
    test.label_for_predicted_variable = label_for_predicted_variable
    test.df = df
    test.start_of_forecast = forecast_period_start_date
    test.period = pd.to_timedelta(no_of_days_in_forecast, unit='d')
    test.end_of_forecast = test.start_of_forecast + test.period
    test.df_train = df[df['ds'] < test.start_of_forecast]
    # check that start_of_prediction_period + prediction_period is not after last date in ds for df
    assert test.end_of_forecast <= get_last_date(df)
    return test


def calc_residuals(test, df_forecast):
    """
    returnerar en dataframe med residualer från prognosen
    """
    df_residuals = pd.DataFrame()
    df_residuals = test.df[(test.df['ds'] >= test.start_of_forecast) &
                            (test.df['ds'] <= test.end_of_forecast)]  # get data for prediction period

    df_residuals.loc[:,'residuals'] = df_forecast['yhat'] - df_residuals['y']  # calc residuals
    df_residuals.loc[:,'residuals_abs'] = abs(df_residuals['residuals'])  # calc absolute residuals
    return df_residuals

import numpy as np

def calc_error_metrics(df_residuals):
    # calc mean absolute error
    mae = df_residuals['residuals_abs'].mean()
  
    # calc RMSE for df_residuals "residuals"
    rmse = np.sqrt(np.mean(df_residuals['residuals']**2))

    # calc error bias
    eb = mae / rmse
    return mae,rmse,eb

def make_forecast_for_test(test: ForecastTest):
    # sourcery skip: inline-immediately-returned-variable
    """ 
    passar df_train till en modell, returnerar ett objekt med prognos, modeell, residualer och beräknat fel
    """
    # Create model
    model = fb.Prophet(changepoint_prior_scale=0.05)
    # Fit model
    model.fit(test.df_train)
    # Make a future dataframe
    future = model.make_future_dataframe(periods=test.period.days)
    # Make predictions
    df_forecast = model.predict(future)
    # cut df_forecast to only contain data from start_of_prediction_period
    df_forecast = df_forecast[df_forecast['ds'] >= test.start_of_forecast]


    # Calculate residuals and error metrics for model forecast
    df_residuals = calc_residuals(test, df_forecast)
    mae,rmse,eb = calc_error_metrics(df_residuals)
    errorMetrics = ErrorMetrics(mae,rmse,eb)

    # Create ForecastResults object for model forecast
    forecast_results = ForecastResults(df_forecast, model, df_residuals, errorMetrics)


    return forecast_results


def make_naive_forecast_for_test(test: ForecastTest):
    """ 
    Gör prognos med en naiv modell (medelvärde för sista perioden i träningsdatat)
    Returnerar ett objekt med prognos, modeell, residualer och beräknat fel
    """
    # add a naive forecast that is sets the predicted value to the mean of the last period in df_train
    df_naive_forecast = pd.DataFrame()
    df_naive_forecast['ds'] = test.df['ds']

    # no of days in test.period
    no_of_days_in_period = test.period.days

    # set mean_val to the mean of 'y' of the last no_of_days_in_period in df_train
    mean_val = test.df_train['y'].tail(n=no_of_days_in_period).mean()

    df_naive_forecast['yhat'] = mean_val
    # cut df_forecast to only contain data from start_of_prediction_period
    df_naive_forecast = df_naive_forecast[df_naive_forecast['ds'] >= test.start_of_forecast]
    # cut df_forecast to only contain data to end_of_prediction_period
    df_naive_forecast = df_naive_forecast[df_naive_forecast['ds'] <= test.end_of_forecast]


    naive_model = None

    # Calculate residuals and error metrics for naive forecast
    df_naive_residuals = calc_residuals(test, df_naive_forecast)
    mae_naive,rmse_naive,eb_naive = calc_error_metrics(df_naive_residuals)

    # errorMetrics = ErrorMetrics(mae,rmse,eb)
    errorMetrics_naive = ErrorMetrics(mae_naive,rmse_naive,eb_naive)
    
    # Create ForecastResults object for naive forecast

    naive_forecast_results = ForecastResults(df_naive_forecast, naive_model, df_naive_residuals, errorMetrics_naive)
    return naive_forecast_results








def run_ForecastTest(test: ForecastTest, naive: bool):
    """
    kör test av forecast enligt parametrar i ForecastTest-objektet
    """
    if naive == False:
        forecast_results = make_forecast_for_test(test)

    else:
        forecast_results = make_naive_forecast_for_test(test)

    plot_result_ForecastTest(test, forecast_results)
    print_dataframe_periods(forecast_results.df_forecast, 'df_naive_forecast')
    print_dataframe_periods(test.df_train, 'df_train')
    print_dataframe_periods(test.df, 'df')
    
    return forecast_results.errorMetrics



def plot_result_ForecastTest(test: ForecastTest, fr: ForecastResults) -> None:
    """
    plottar resultatet av testet som en graf med prognos och utfall
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # add title to plot
    fig.suptitle(test.label_for_predicted_variable, fontsize=16)
    # add label to x-axis
    ax.set_xlabel('Datum', fontsize=14)
    # add label to y-axis
    ax.set_ylabel(test.label_for_predicted_variable, fontsize=14)
    fr.df_forecast.plot(x='ds', y='yhat', ax=ax, label='prognos')

    test.df[(test.df['ds'] >= test.start_of_forecast - test.period) & 
        (test.df['ds'] < test.start_of_forecast + test.period)].plot(x='ds', y='y', ax=ax, label='utfall')    
    test.df_train[(test.df_train['ds'] >= test.start_of_forecast - test.period)].plot(x='ds', y='y', ax=ax, label='träning')

    ax.annotate(f'Mean abs error: {fr.errorMetrics.MAE:.1f}', xy=(0.80, 0.15), xycoords='axes fraction')
    ax.annotate(f'RMSE: {fr.errorMetrics.RMSE:.1f}', xy=(0.80, 0.10), xycoords='axes fraction')
    ax.annotate(f'EB: {fr.errorMetrics.EB:.1f}', xy=(0.80, 0.05), xycoords='axes fraction')
    
    # add legend
    ax.legend()
    # show plot
    
            
    plt.show()
    # plot model components
    if fr.model is not None:
        fr.model.plot_components(fr.df_forecast)
        plt.show()


def get_last_date(df: pd.DataFrame) -> pd.Timestamp:
    return df.iloc[-1]['ds']

def get_first_date(df: pd.DataFrame) -> pd.Timestamp:
    return df.iloc[0]['ds']

def print_dataframe_periods(df: pd.DataFrame, label: str) -> None:
    print(f'period of {label}: {get_first_date(df).date()} to {get_last_date(df).date()}')


