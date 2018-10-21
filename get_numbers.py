
import sys
import itertools
import numpy as np
import random

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import signal
import time

import warnings

warnings.filterwarnings('ignore')

my_random_value = 95
my_index = 8 # strat from 0
distance_judgement_threshold = 30
var_judgement_threshold = 200
epochs_to_estimate_random = 80
first_predict_value_1 = 12
first_predict_value_2 = 6
history_thre = -25
p_order=(0,1,2,3)
q_order=(0,1,2,3)
numbers_to_mean=-1


from threading import Thread
import functools

def timeout(timeout):
    def deco(func): 		
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                #print 'error starting thread'
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco

def LineToNums(line, type=float):
	"""将输入的行按间隔符分割，并转换为某种数字类型的可迭代对象（默认为float类型）"""
	return (type(cell) for cell in line.split('\t'))

metaLine = sys.stdin.readline()
lineNum, columnNum = LineToNums(metaLine, int) # ???

history = []
#  get the history information, history[i][0], history[i][1][0] and history[i][1][2], i is the number of epoch
for line in map(lambda _: sys.stdin.readline(), range(lineNum)):
	gnum, *nums = LineToNums(line)
	history.append((gnum, nums))

def Mean(iter, len):
	"""用于计算均值的帮主函数"""
	return sum(iter) / len

def data_transfer(history):
	new_history = history.copy()
	for i in range(len(history)): #可能这个循环不需要,只要把i=-1带入下面的语句???
		golden_num = new_history[i][0]
		for x in range(int(len(history[i][1])/2)):
			if x != my_index:
				if abs(new_history[i][1][2*x+0]-golden_num) >= abs(new_history[i][1][2*x+1]-golden_num): # 交换后, 大的数在右边, 小的数在左边
					tmp = new_history[i][1][2*x+0]
					new_history[i][1][2*x+0] = new_history[i][1][2*x+1]
					new_history[i][1][2*x+1] = tmp
	return new_history

@timeout(4.5)
def my_ARIMA(new_history, aaa):# 输出 1+2n长度的一个list, 第一个值和G有关, output: my_ARIMA_array
	GD_history=[]
	for i in range(len(new_history)):
		tmp = new_history[i][0]*len(new_history[i][1])/0.618-(new_history[i][1][2*my_index]+new_history[i][1][2*my_index+1])
		GD_history.append( tmp/(len(new_history[i][1])-2 ) *0.618   )
	timeseries = pd.DataFrame(GD_history)[history_thre:]
	min_ave_rss = 10000
	for p in p_order:
		for q in q_order:
			if p == q == 0:
				continue
			model = ARIMA(timeseries, order=(p, 0, q))  
			try:
				results_AR = model.fit(disp=-1)
			except:
				continue
			else:
				ave_rss=sum((results_AR.fittedvalues-timeseries)**2)/len(timeseries)
				if ave_rss < min_ave_rss:
					min_ave_rss = ave_rss
					p_best=p
					q_best=q
					results_AR_best = results_AR
	if min_ave_rss == 10000:
		forecast = GD_history[-1]
	else:
		forecast = results_AR_best.forecast()[0]
	ccc=aaa.copy()
	for x in range(len(ccc)):
		if ccc[x]==-1:
			ccc[x]=forecast
	return ccc

# 输出[-1, -1, [随机数均值, 随机概率], -1, [随机数均值, 随机概率], -1, -1] 最后一个-1表示最后一个bot没有捣乱, 对我们的数据列，填充-1
def detect_random(history):
	out_array = [] 
	out_array.append(-1) # golden num position
	for x in range(int(len(history[0][1])/2)):
		column_value = []
		column_random_value = []
		column_normal_value = []
		random_num = 0
		out_array.append(-1) # 每一组左边那列一定是正常的, 填充-1
		for i in range(len(history)):	
			column_value.append(history[i][1][2*x+1])
			#if abs(history[i][1][2*x+1] - my_ARIMA_array[2*x+1]) > distance_judgement_threshold:
			if abs(history[i][1][2*x+1] - history[i][1][2*x]) > distance_judgement_threshold:
				random_num += 1
				column_random_value.append(history[i][1][2*x+1])
			else:
				column_normal_value.append(history[i][1][2*x+1])
		
		if np.var(column_value)<=var_judgement_threshold or x==my_index: #if random_num==0:自己那两列认为没有random，不然后续还要处理
			out_array.append(-1)
		else:
			ratio = random_num / len(history) 
			if len(column_random_value)==0:
				out_array.append(-1)
			else:
				avg = np.mean(column_random_value) # 随机数的均值
				random_info_k = [avg, ratio]
				out_array.append(random_info_k)
	return out_array

def predict(info_array):
	random_prob = []
	random_value = []
	sum_all = 0 # 除我们预测的两个数外的所有其他Bot预测的值的和
	for x in range(len(info_array)):
		if isinstance(info_array[x], list): 
			random_prob.append(info_array[x][1]) # 捣乱Bot的随机概率
			random_value.append([info_array[x][0],info_array[x-1]]) # # 捣乱Bot的随机数均值, 以及ARIMA预测的该bot未捣乱时的预测值
		elif (x!=0) and (isinstance(info_array[x], float)==True) and (x!=2*my_index+1) and (x!=2*my_index+2): # ARIMA模型的输出, 第一列是G值, 不用加, 自己的那两列也不用加, 捣乱的那些也不用ARIMA加
			sum_all += info_array[x]
		else:
			pass
	
	prob_array = [0] * (2^len(random_prob))
	value_array = [0] * (2^len(random_prob))
	for i in range(2^len(random_prob)): # 只考虑了会产生随机数的那些列
		bi_index = '{:016b}'.format(i) # 假设应该不会超过16个bot
		bi_index = bi_index[-len(random_prob):] # str: '0001001101', 截取
		prob_tmp = 1
		value_tmp = 0
		for i in range(len(random_prob)):
			prob_tmp *= ( int(bi_index[i])*random_prob[i] + (1-int(bi_index[i]))*(1-random_prob[i]) )
			value_tmp += ( int(bi_index[i])*random_value[i][0] + (1-int(bi_index[i]))*random_value[i][1] )
		prob_array.append(prob_tmp)
		value_array.append(value_tmp)
	my_dict = [(prob_array[i], value_array[i]) for i in range(len(prob_array))]
	my_dict.sort(key=lambda elem: elem[0])

	if len(my_dict)==0:
		sum_all +=0
	else:
		mean_list=[]
		for item in my_dict[numbers_to_mean:]:
			mean_list.append(item[1])
		avg = np.mean(mean_list)
		#avg = np.mean(my_dict[-1:][1]) #
		sum_all += avg

	my_value_1 = history[-1][1][2*my_index]
	my_value_2 = history[-1][1][2*my_index+1]
	if my_value_1 == my_random_value:
		my_arima_value = my_value_2
	else:
		my_arima_value = my_value_1
	
	#scale_parameter = max(min(1 * history[-1][0] / my_arima_value, 1.3), 0.6)
	scale_parameter =  history[-1][0] / my_arima_value
	if len(history)>100 and len(history)<175:
		my_random_probability = 0.1
	elif len(history)>275 and len(history)<350:
		my_random_probability = 0.1
	else:
		my_random_probability = 0
	if random.random() < my_random_probability:
		predict_value_2 = my_random_value
		predict_value_1 = 0.97*(sum_all+predict_value_2) / (len(history[0][1])-0.97)
	else:
		predict_value_1 = 0.97*sum_all / (len(history[0][1])-0.97*(1+scale_parameter))
		predict_value_2 = scale_parameter * predict_value_1

	return predict_value_1, predict_value_2

if len(history) == 0:
	print("%f\t%f" % (first_predict_value_1, first_predict_value_2))
elif len(history) == 1:
	print("%f\t%f" % (history[-1][0]*0.62, history[-1][0]*0.7))
elif len(history) == 2:
	print("%f\t%f" % (history[-1][0]*0.7, history[-1][0]*0.8))
elif len(history) == 3:
	print("%f\t%f" % (history[-1][0]*0.8, history[-1][0]*0.9))
elif len(history)<=30:
	# 取最近的记录，最多五项。计算这些记录中黄金点的均值作为本脚本的输出。
	candidate1 = Mean(map(lambda h: h[0], history[-5:]), min(len(history), 5))
	candidate2 = candidate1 * 0.92 # 第二输出。
	candidate3 = candidate1 * 0.82 # 第三输出。
	print("%f\t%f" % (candidate2, candidate3))

else:
	
	new_history = data_transfer(history)
	
	info_array_2 = detect_random(history[-epochs_to_estimate_random:])
	time_out_signal = False
	try:
		info_array_1 = my_ARIMA(new_history, info_array_2)
	except:
		time_out_signal = True

	if time_out_signal:
		candidate1 = Mean(map(lambda h: h[0], history[-5:]), min(len(history), 5))
		candidate2 = candidate1 * 0.92 # 第二输出。
		candidate3 = candidate1 * 0.82 # 第三输出。
		print("%f\t%f" % (candidate2, candidate3))
	else:
		info_array_3 = []
		for x in range(len(info_array_2)):
			if info_array_2[x]!=-1:
				info_array_3.append(info_array_2[x])
			else:
				info_array_3.append(info_array_1[x][0])
		
		predict_value_1, predict_value_2 = predict(info_array_3)

		if predict_value_2==my_random_value:
			if random.random() > 0.5:
				print("%f\t%f" % (predict_value_1, predict_value_2))
			else:
				print("%f\t%f" % (predict_value_2, predict_value_1))
		else:
			print("%f\t%f" % (predict_value_1, predict_value_2))
		
