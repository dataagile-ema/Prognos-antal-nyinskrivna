import pandas as pd
import fbprophet as fb
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class ForecastTest:
    """"
    Parametrar för att göra ett test av prognos av antal nyinskrivna på IVA.
    start_of_prediction: datum då prognosen ska börja
    end_of_prediction: datum då prognosen ska slutar
    period: antal dagar som ska prognoseras
    df_train: dataframe med data som modellen passas med
    df: hela datasetet som läses in från csb-filen, som testet använder som träningsdata och utfall.
    """
    start_of_prediction: pd.Timestamp 
    end_of_prediction: pd.Timestamp
    period: pd.Timedelta
    df_train: pd.DataFrame
    df: pd.DataFrame


def read_and_prep_data(path: str):
    """
    läser csv fil med format ds (datum) och y (antal nyinskrivna)
    """
    df = pd.read_csv(path)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values(by='ds')
    return df


def create_ForecastTest(df: pd.DataFrame, pred_period_start_date_str, prediction_period_str):
    """
    Skapar ett objekt med paramterar för ett testa prognos av antal nyinskrivna på IVA
    df: hela datasetet som läses in från csb-filen, som testet använder som träningsdata och utfall.
    pred_period_start_date_str: datum då prognosen ska börja (exemple: "2021-01-01")
    prediction_period_str: antal dagar som ska prognoseras (exemple: "15 days)
    """
    test = ForecastTest(None, None, None, None, None)
    test.df = df
    test.start_of_prediction = pd.to_datetime(pred_period_start_date_str)
    test.period = pd.to_timedelta(prediction_period_str)
    test.end_of_prediction = test.start_of_prediction + test.period
    test.df_train = df[df['ds'] < test.start_of_prediction]
    # check that start_of_prediction_period + prediction_period is not after last date in ds for df
    assert test.end_of_prediction <= get_last_date(df)
    return test


def make_forecast_for_test(test: ForecastTest):
    """ 
    passar df_train till en modell, returnerar en prognos från modellen tillsammans med modell-objektet
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
    df_forecast = df_forecast[df_forecast['ds'] >= test.start_of_prediction]
    return model,df_forecast


def run_ForecastTest(test) -> None:
    """
    kör test av forecast enligt parametrar i ForecastTest-objektet
    """
    model, df_forecast = make_forecast_for_test(test)
    plot_result_ForecastTest(test, df_forecast, model)

    print_dataframe_periods(df_forecast, 'df_forecast')
    print_dataframe_periods(test.df_train, 'df_train')
    print_dataframe_periods(test.df, 'df')


def plot_result_ForecastTest(test: ForecastTest, df_forecast: pd.DataFrame, model: fb.Prophet):
    """
    plottar resultatet av testet som en graf med prognos och utfall
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # add title to plot
    fig.suptitle('Antal nyinskrivna', fontsize=16)
    # add label to x-axis
    ax.set_xlabel('Datum', fontsize=14)
    # add label to y-axis
    ax.set_ylabel('Antal nyinskrivna', fontsize=14)
    df_forecast.plot(x='ds', y='yhat', ax=ax, label='prognos')

    test.df[(test.df['ds'] >= test.start_of_prediction - test.period) & 
        (test.df['ds'] < test.start_of_prediction + test.period)].plot(x='ds', y='y', ax=ax, label='utfall')    
    test.df_train[(test.df_train['ds'] >= test.start_of_prediction - test.period) & 
        (test.df_train['ds'] < test.start_of_prediction + test.period)].plot(x='ds', y='y', ax=ax, label='träning')
    plt.show()
    # plot model components
    model.plot_components(df_forecast)
    plt.show()


def get_last_date(df: pd.DataFrame) -> pd.Timestamp:
    return df.iloc[-1]['ds']

def get_first_date(df: pd.DataFrame) -> pd.Timestamp:
    return df.iloc[0]['ds']

def print_dataframe_periods(df: pd.DataFrame, label: str) -> None:
    print(f'period of {label}: {get_first_date(df).date()} to {get_last_date(df).date()}')


