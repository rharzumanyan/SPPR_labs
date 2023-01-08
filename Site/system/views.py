from django.shortcuts import render
from django.http import HttpResponse
from .models import Unit
from .forms import UnitForm

import os
import random
import numpy as np
import pandas as pd
from scipy import sparse
import lightfm
from lightfm import LightFM, cross_validation
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def create_interaction_matrix(df,user_col, item_col, rating_col, threshold = None):

	interactions = df.groupby([user_col, item_col])[rating_col] \
			.sum().unstack().reset_index(). \
			fillna(0).set_index(user_col)

	return interactions

def random_10():
	with open('media/user_dict.pkl', 'rb') as f:
		user_dict = pickle.load(f)
	with open('media/item_dict.pkl', 'rb') as f:
		item_dict = pickle.load(f)
	df_playlist=pd.read_csv('media/df_playlist.csv', sep='\t')
	interactions = create_interaction_matrix(df = df_playlist, user_col = "user_id", item_col = 'cellphone_id', rating_col = 'rating', threshold = None)
	nrec_items = 10
	top_id = random.sample(list(user_dict.keys()), 10)
	filename = 'media/finalized_model.sav'
	model = pickle.load(open(filename, 'rb'))
	threshold = 0
	n_users, n_items = interactions.shape
	show = True
	rez = []
	for user_id in top_id:
		user_x = user_dict[user_id]
		scores = pd.Series(model.predict(user_x,np.arange(n_items)))
		scores.index = interactions.columns
		scores = list(pd.Series(scores.sort_values(ascending=False).index))

		known_items = list(pd.Series(interactions.loc[user_id,:] \
									 [interactions.loc[user_id,:] > threshold].index) \
						   .sort_values(ascending=False))

		scores = [x for x in scores if x not in known_items]
		return_score_list = scores[0:nrec_items]
		known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
		scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
		if show == True:
			for i in known_items:
				rez.append((str(user_id) + '- ' + i))
	return rez


def sample_recommendation_user(user_id):
	with open('media/user_dict.pkl', 'rb') as f:
		user_dict = pickle.load(f)
	with open('media/item_dict.pkl', 'rb') as f:
		item_dict = pickle.load(f)
	df_playlist=pd.read_csv('media/df_playlist.csv', sep='\t')
	interactions = create_interaction_matrix(df = df_playlist, user_col = "user_id", item_col = 'cellphone_id', rating_col = 'rating', threshold = None)
	nrec_items = 10
	
	filename = 'media/finalized_model.sav'
	model = pickle.load(open(filename, 'rb'))
	threshold = 0
	n_users, n_items = interactions.shape
	show = True
	user_x = user_dict[user_id]
	scores = pd.Series(model.predict(user_x,np.arange(n_items)))
	scores.index = interactions.columns
	scores = list(pd.Series(scores.sort_values(ascending=False).index))
	
	known_items = list(pd.Series(interactions.loc[user_id,:] \
								 [interactions.loc[user_id,:] > threshold].index) \
					   .sort_values(ascending=False))
	
	scores = [x for x in scores if x not in known_items]
	return_score_list = scores[0:nrec_items]
	known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
	scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
	rez=[]
	if show == True:
		counter = 1
		for i in scores:
			rez.append((str(counter) + '- ' + i))
			counter+=1
	return rez

def index(request):
	return render(request, 'system/index.html')
	

def recom(request):
	work = Unit.objects.order_by('-user_id')
	if work:
		for n in work:
			if n.recomendations=="NULL":
				n.recomendations = sample_recommendation_user(n.user_id)
				n.save()
	return render(request, 'system/recom.html', {'work':work})
	

def post(request):
	ex = random_10()
	error = ''
	if request.method == 'POST':
		form = UnitForm(request.POST)

		if form.is_valid():
			form.save()
		else:
			error = 'Ошибка заполнения'
			
	form = UnitForm()
	data = {
		'form': form,
		'error': error,
		'ex': ex
	}
	return render(request, 'system/post.html', data)
