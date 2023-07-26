import os
import time
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter

import wandb
import model
import evaluate
import data_utils
from loss import loss_function, loss_function_co_teaching

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
	type = str,
	help = 'dataset used for training, options: amazon_book, yelp, adressa',
	default = 'adressa')
parser.add_argument('--model_1', 
	type = str,
	help = 'model used for training. options: GMF, NeuMF-end',
	default = 'GMF')
parser.add_argument('--model_2', 
	type = str,
	help = 'model used for training. options: GMF, NeuMF-end',
	default = 'GMF')
parser.add_argument('--drop_rate', 
	type = float,
	help = 'drop rate',
	default = 0.2)
parser.add_argument('--num_gradual', 
	type = int, 
	default = 30000,
	help='how many epochs to linearly increase drop_rate')
parser.add_argument('--exponent', 
	type = float, 
	default = 1, 
	help='exponent of the drop rate {0.5, 1, 2}')
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=1024, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=10,
	help="training epoches")
parser.add_argument("--eval_freq", 
	type=int,
	default=2000,
	help="the freq of eval")
parser.add_argument("--top_k", 
	type=list, 
	default=[3, 20, 50, 100],
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
	type=int,
	default=3, 
	help="number of layers in MLP model")
parser.add_argument("--num_ng", 
	type=int,
	default=1, 
	help="sample negative items for training")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="1",
	help="gpu card ID")
args = parser.parse_args()
wandb.login
wandb.init(project="Co-teaching", config=args, notes=args.model_1+args.model_2)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

torch.manual_seed(2019) # cpu
torch.cuda.manual_seed(2019) #gpu
np.random.seed(2019) #numpy
# random.seed(2019) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn
random_number = random.randint(10000, 99999)

def worker_init_fn(worker_id):
    np.random.seed(2019 + worker_id)

data_path = '../data/{}/'.format(args.dataset)
model_path = '/home/hezhuangzhuang/Code_Label_Noise/Coteaching-Rec/T_CE/models/{}/'.format(args.dataset)
print("arguments: %s " %(args))
print("config model_1", args.model_1)
print("config model_2", args.model_2)
print("config data path", data_path)
print("config model path", model_path)

############################## PREPARE DATASET ##########################

train_data, valid_data, test_data_pos, user_pos, user_num ,item_num, train_mat, train_data_noisy = data_utils.load_all(args.dataset, data_path)

# construct the train and test datasets
train_dataset = data_utils.NCFData(
		train_data, item_num, train_mat, args.num_ng, 0, train_data_noisy)
valid_dataset = data_utils.NCFData(
		valid_data, item_num, train_mat, args.num_ng, 1)

train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
valid_loader = data.DataLoader(valid_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

print("data loaded! user_num:{}, item_num:{} train_data_len:{} test_user_num:{}".format(user_num, item_num, len(train_data), len(test_data_pos)))

########################### CREATE MODEL #################################
if args.model_1 == 'NeuMF-pre': # pre-training. Not used in our work.
	GMF_model_path = model_path + 'GMF.pth'
	MLP_model_path = model_path + 'MLP.pth'
	NeuMF_model_path = model_path + 'NeuMF.pth'
	assert os.path.exists(GMF_model_path), 'lack of GMF model'
	assert os.path.exists(MLP_model_path), 'lack of MLP model'
	GMF_model = torch.load(GMF_model_path)
	MLP_model = torch.load(MLP_model_path)
else:
	GMF_model = None
	MLP_model = None

model_1 = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
						args.dropout, args.model_1, GMF_model, MLP_model)
model_2 = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
						args.dropout, args.model_2, GMF_model, MLP_model)
model_1.cuda()
model_2.cuda()
BCE_loss = nn.BCEWithLogitsLoss()

if args.model_1 == 'NeuMF-pre':
	optimizer_1 = optim.SGD(model_1.parameters(), lr=args.lr)
	optimizer_2 = optim.SGD(model_2.parameters(), lr=args.lr)
else:
	optimizer_1 = optim.Adam(model_1.parameters(), lr=args.lr)
	optimizer_2 = optim.Adam(model_2.parameters(), lr=args.lr)

# writer = SummaryWriter() # for visualization

# define drop rate schedule
def drop_rate_schedule(iteration):

	drop_rate = np.linspace(0, args.drop_rate**args.exponent, args.num_gradual)
	if iteration < args.num_gradual:
		return drop_rate[iteration]
	else:
		return args.drop_rate


########################### Eval #####################################
def eval(which_model, model, valid_loader, best_loss, count, model_name):
    	
    model.eval()
    epoch_loss = 0
    valid_loader.dataset.ng_sample() # negative sampling
    for user, item, label, noisy_or_not in valid_loader:
        user = user.cuda()
        item = item.cuda()
        label = label.float().cuda()

        prediction = model(user, item)
        # loss = BCE_loss(prediction, label)
        loss = loss_function(prediction, label, drop_rate_schedule(count))
        epoch_loss += loss.detach()
    print("################### EVAL ######################")
    print("Eval loss:{}".format(epoch_loss))
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        if args.out:
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            torch.save(model, '{}{}_{}-{}-{}-{}.pth'.format(model_path, model_name, args.drop_rate, args.num_gradual, which_model, random_number))
    return best_loss

########################### Test #####################################
def test(model, test_data_pos, user_pos):
	top_k = args.top_k
	model.eval()
	_, recall, NDCG, _ = evaluate.test_all_users(model, 4096, item_num, test_data_pos, user_pos, top_k)

	print("################### TEST ######################")
	print("Recall {:.4f}-{:.4f}-{:.4f}-{:.4f}".format(recall[0], recall[1], recall[2], recall[3]))
	print("NDCG {:.4f}-{:.4f}-{:.4f}-{:.4f}".format(NDCG[0], NDCG[1], NDCG[2], NDCG[3]))
	wandb.log({
	"Recall@3": recall[0],
	"Recall@20": recall[1],
	"Recall@50": recall[2],
	"Recall@100": recall[3],
	"NDCG@3": NDCG[0],
	"NDCG@20": NDCG[1],
	"NDCG@50": NDCG[2],
	"NDCG@100": NDCG[3],})
########################### TRAINING #####################################
count, best_hr = 0, 0
best_loss_1 = 1e9
best_loss_2 = 1e9

for epoch in range(args.epochs):
	model_1.train() # Enable dropout (if have).
	model_2.train() # Enable dropout (if have).
	start_time = time.time()
	train_loader.dataset.ng_sample()

	for user, item, label, noisy_or_not in train_loader:
		user = user.cuda()
		item = item.cuda()
		label = label.float().cuda()

		model_1.zero_grad()
		model_2.zero_grad()
		prediction_1 = model_1(user, item)
		prediction_2 = model_2(user, item)
		loss_1,loss_2 = loss_function_co_teaching(prediction_1, prediction_2, label, drop_rate_schedule(count))
		loss_1.backward()
		loss_2.backward()
		optimizer_1.step()
		optimizer_2.step()
		wandb.log({'drop_rate' : drop_rate_schedule(count)})
		if count % args.eval_freq == 0 and count != 0:
			print("epoch: {}, iter: {}, loss:{}".format(epoch, count, loss_1))
			best_loss_1 = eval('model_1', model_1, valid_loader, best_loss_1, count, args.model_1)
			print("epoch: {}, iter: {}, loss:{}".format(epoch, count, loss_2))
			best_loss_2 = eval('model_2', model_2, valid_loader, best_loss_2, count, args.model_2)
			model_1.train()
			model_2.train()
			wandb.log({
			"epoch": epoch,
			"iter": count,
			"loss_1": loss_1,
			"loss_2": loss_2})

		count += 1

print("############################## Training End. ##############################")
test_model_1 = torch.load('{}{}_{}-{}-{}-{}.pth'.format(model_path, args.model_1, args.drop_rate, args.num_gradual, 'model_1', random_number))
test_model_1.cuda()
test(test_model_1, test_data_pos, user_pos)
test_model_2 = torch.load('{}{}_{}-{}-{}-{}.pth'.format(model_path, args.model_1, args.drop_rate, args.num_gradual, 'model_1', random_number))
test_model_2.cuda()
test(test_model_2, test_data_pos, user_pos)
# test(model_2, test_data_pos, user_pos)