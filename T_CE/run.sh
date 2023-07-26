# nohup python -u main.py --dataset=$1 --model=$2 --drop_rate=$3 --num_gradual=$4 --gpu=$5 > log/$1/$2_$3-$4.log 2>&1 &
python main.py --dataset=adressa --model_1=GMF --model_2=NeuMF-end
python main.py --dataset=yelp --model_1=GMF --model_2=NeuMF-end
python main.py --dataset=amazon_book --model_1=GMF --model_2=NeuMF-end