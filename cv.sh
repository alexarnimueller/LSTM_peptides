python LSTM_peptides.py --run_name cv_l1_n64 --neurons 64 --epochs 75 --sample 100 --layers 1 --dropout 0.2 --lr 0.01 --temp 1.2 --cv 10 --valsplit 0.0
sleep 1
python LSTM_peptides.py --run_name cv_l1_n128 --neurons 128 --epochs 75 --sample 100 --layers 1 --dropout 0.2 --lr 0.01 --temp 1.2 --cv 10 --valsplit 0.0
sleep 1
python LSTM_peptides.py --run_name cv_l1_n256 --neurons 256 --epochs 75 --sample 100 --layers 1 --dropout 0.2 --lr 0.01 --temp 1.2 --cv 10 --valsplit 0.0
sleep 1
python LSTM_peptides.py --run_name cv_l1_n512 --neurons 512 --epochs 75 --sample 100 --layers 1 --dropout 0.2 --lr 0.01 --temp 1.2 --cv 10 --valsplit 0.0
sleep 1
python LSTM_peptides.py --run_name cv_l1_n1024 --neurons 1024 --epochs 75 --sample 100 --layers 1 --dropout 0.2 --lr 0.01 --temp 1.2 --cv 10 --valsplit 0.0
sleep 1
python LSTM_peptides.py --run_name cv_l2_n64 --neurons 64 --epochs 75 --sample 100 --layers 2 --dropout 0.2 --lr 0.01 --temp 1.2 --cv 10 --valsplit 0.0
sleep 1
ppython LSTM_peptides.py --run_name cv_l2_n128 --neurons 128 --epochs 75 --sample 100 --layers 2 --dropout 0.2 --lr 0.01 --temp 1.2 --cv 10 --valsplit 0.0
sleep 1
ppython LSTM_peptides.py --run_name cv_l2_n256 --neurons 256 --epochs 75 --sample 100 --layers 2 --dropout 0.2 --lr 0.01 --temp 1.2 --cv 10 --valsplit 0.0
sleep 1
ppython LSTM_peptides.py --run_name cv_l2_n512 --neurons 512 --epochs 75 --sample 100 --layers 2 --dropout 0.2 --lr 0.01 --temp 1.2 --cv 10 --valsplit 0.0
sleep 1
ppython LSTM_peptides.py --run_name cv_l2_n1024 --neurons 1024 --epochs 75 --sample 100 --layers 2 --dropout 0.2 --lr 0.01 --temp 1.2 --cv 10 --valsplit 0.0
sleep 1
python LSTM_peptides.py --run_name cv_l3_n64 --neurons 64 --epochs 75 --sample 100 --layers 3 --dropout 0.2 --lr 0.01 --temp 1.2 --cv 10 --valsplit 0.0
sleep 1
python LSTM_peptides.py --run_name cv_l3_n128 --neurons 128 --epochs 75 --sample 100 --layers 3 --dropout 0.2 --lr 0.01 --temp 1.2 --cv 10 --valsplit 0.0
sleep 1
python LSTM_peptides.py --run_name cv_l3_n256 --neurons 256 --epochs 75 --sample 100 --layers 3 --dropout 0.2 --lr 0.01 --temp 1.2 --cv 10 --valsplit 0.0
sleep 1
python LSTM_peptides.py --run_name cv_l3_n512 --neurons 512 --epochs 75 --sample 100 --layers 3 --dropout 0.2 --lr 0.01 --temp 1.2 --cv 10 --valsplit 0.0
sleep 1
python LSTM_peptides.py --run_name cv_l3_n1024 --neurons 1024 --epochs 75 --sample 100 --layers 3 --dropout 0.2 --lr 0.01 --temp 1.2 --cv 10 --valsplit 0.0
