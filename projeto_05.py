import yfinance as yf

ticker = input("Digite o código da ação: ")
dados = yf.Ticker(ticker).history("2y")
dados.head

dados["Close"]

treinamento = dados.reset_index()
treinamento = treinamento[["Date", "Close"]]
treinamento["Date"].dt.tz_localize(None)
treinamento["Date"] = treinamento["Date"].dt.tz_localize(None)
treinamento.columns = ['ds', 'y']

from prophet import Prophet
from prophet.plot import plot_plotly

modelo = Prophet()
modelo.fit(treinamento)

periodo = modelo.make_future_dataframe(90)
previsoes = modelo.predict(periodo)

plot_plotly(modelo, previsoes)