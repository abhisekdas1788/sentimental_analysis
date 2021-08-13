from django.shortcuts import render
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import nltk
nltk.download('punkt')

#xgb_model = pickle.load(open('./models/xgbclassif_2.pkl', 'rb'))
#rf_model = pickle.load(open('./models/rfclassif_2.pkl', 'rb'))
lr_model = pickle.load(open('./models/logisticclassif_2.pkl', 'rb'))
vect = pickle.load(open('./models/vect.pkl', 'rb'))

def index(request):
    return render(request, 'index.html')


def predictSenti(request):
    print (request)
    if request.method == 'POST':
        review = request.POST.get('review')
    print ('review', review)

    textt = vect.transform(sent_tokenize(review))
    #rating_xgb = int(xgb_model.predict(textt))
    #rating_rf = int(rf_model.predict(textt))
    rating_lr = int(lr_model.predict(textt))

    print ('rating', rating_lr)
    context = {'review': review, 'rating_lr': rating_lr}
    return render(request, 'index.html', context)