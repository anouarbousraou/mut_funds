import json
import elemental
import yfinance
import urllib.parse as urlparse
from urllib.parse import parse_qs

def get_quote(symbol):
    msft = yfinance.Ticker(symbol)
    try:
        hist = msft.history(period="2d")
    except json.decoder.JSONDecodeError:
        return None
    try:
        hist.reset_index(inplace=True)
        jsdata = json.loads(hist.to_json())
        return jsdata["Close"]["0"]
    except (ValueError, KeyError) as e:
       return None


def web_lookup(browser, isin):
    # Search PyPI for Elemental.
    browser = elemental.Browser()
    browser.visit("https://finance.yahoo.com/lookup")
    browser.get_input(id="yfin-usr-qry").fill(isin)
    browser.get_button(type="submit").click()

    time.sleep(5)

    parsed = urlparse.urlparse(browser.url)
    try:
        ticker = parse_qs(parsed.query)['p'][0]
    except KeyError:
        ticker = "n/a"
    browser.quit()
    return ticker