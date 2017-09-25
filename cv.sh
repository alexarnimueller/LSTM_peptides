python LSTM_peptides.py --run_name cv_w30 --valsplit 0.0 --epochs 30 --cv 5 --window 30
sleep 1
python LSTM_peptides.py --run_name cv_w36 --valsplit 0.0 --epochs 30 --cv 5 --window 36
sleep 1
python LSTM_peptides.py --run_name cv_w40 --valsplit 0.0 --epochs 30 --cv 5 --window 40
