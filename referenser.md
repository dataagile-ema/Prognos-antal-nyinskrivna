## lägga till miljö
https://facebook.github.io/prophet/
conda create --name time_series_test python=3.9
conda activate time_series_test
conda install -c conda-forge prophet
conda install -c conda-forge imageio
conda install -c anaconda ipykernel

## ta bort miljö 
conda env remove -n time_series_test

## installera neural prophet
https://github.com/ourownstory/neural_prophet
conda create --name time_series_test_n python=3.9
conda activate time_series_test_n
conda install -c anaconda ipykernel
pip install neuralprophet
conda install -c conda-forge imageio


##
pip install neuralprophet[live]