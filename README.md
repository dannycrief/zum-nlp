# ZUM NLP Project
## Preinstall steps

- Create virtual environment

```shell
python3 -m venv venv
```
- Install requirements

```shell
source venv/bin/activate && pip install -r requirements.txt
```

- Add your credentials to .env file
```shell
cat > .env << EOF
CLIENT_ID=<client-id>
CLIENT_SECRET=<client-secret>
USER_AGENT=python:nlp_zum:v1.0 (by /u/dancrief)
EOF
```

## Etap 1
To run first etap you should run [main.py](main.py) file
## Etap 2
Etap 2 located at sentiment_analysis/[etap2_sentiment_analysis_classic_ml.py](sentiment_analysis%2Fetap2_sentiment_analysis_classic_ml.py)
## Etap 3
Etap 3 located at sentiment_analysis/[etap3_sentiment_analysis_lstm_model.py](sentiment_analysis%2Fetap3_sentiment_analysis_lstm_model.py)
## Etap 4
Etap 4 located at sentiment_analysis/[etap4.py](sentiment_analysis%2Fetap4.py)
