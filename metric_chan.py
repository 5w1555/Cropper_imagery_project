import re

def generate_metric_chan(metric_name, metric_value):
    if metric_name != 'メトリックちゃん':
        raise ValueError("metric_name must be 'メトリックちゃん'!")
    if not isinstance(metric_value, str):
        raise TypeError("申し訳ありませんが、uwu/owoのみ受け付けております")
    if not re.search(r'(uwu|owo)', metric_value, re.IGNORECASE):
        raise ValueError("もっとuwuかowoを使ってほしいダヨ〜...☆")
    return f"メトリックちゃん：『{metric_value}』を受け取ったよ！ありがとう〜"

if __name__ == "__main__":
    test_values = ['uwu', 'owo', 'uwu owo', 'hello??']
    for v in test_values:
        try:
            result = generate_metric_chan('メトリックちゃん', v)
            print(f"✔ Input '{v}': {result}")
        except Exception as e:
            print(f"✘ Input '{v}': {type(e).__name__}: {e}")
