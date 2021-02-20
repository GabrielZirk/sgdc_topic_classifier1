import flask
from nltk.corpus import stopwords
from HanTa import HanoverTagger as ht
from sklearn.linear_model import SGDClassifier
import joblib
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
nltk.download("stopwords")
import os

model_dbow = Doc2Vec.load("model/classifier_d2v_gensimtutorial.npy")
sgdc = joblib.load("model/SGDCClassifier.pkl")

app = flask.Flask(__name__, template_folder = "templates")

@app.route("/", methods = ["GET", "POST"])
def topic_prediction():

    if flask.request.method == "GET":
        return(flask.render_template("main.html"))

    if flask.request.method == "POST":

        text_to_predict = flask.request.form["article_input"]


        tokenizer = nltk.RegexpTokenizer(r"\w+")
        tokenized = tokenizer.tokenize(text_to_predict)
        # print("Tokens :2", tokenized[:10])

        german_stop_words = stopwords.words('german')
        wo_stopwords = [word for word in tokenized if word not in german_stop_words]

        tagger = ht.HanoverTagger('morphmodel_ger.pgz')
        lemmatized = [lemma for (word, lemma, pos) in tagger.tag_sent(wo_stopwords)]
        # print("Lemma : ",lemmatized[:10])

        vectorized = model_dbow.infer_vector(lemmatized).reshape(1, -1)
        # print("Vector : ", vectorized)

        prediction = sgdc.predict(vectorized)
        # print("Prediction : ", prediction)

        topic_code = {
            0: "Kultur",
            1: "Wissenschaft",
            2: "Etat",
            3: "Inland",
            4: "Sport",
            5: "Wirtschaft",
            6: "International",
            7: "Web",
            8: "Panorama"
        }

        # return topic_code[int(prediction)]




        return flask.render_template("main.html", original_input = text_to_predict, result = topic_code[int(prediction)])

if __name__ == '__main__':
    app.run(debug = True, host = "127.0.0.1", port = "8080")
    # port = int(os.environ.get("PORT", 5000))
    # app.run(debug = True, host='0.0.0.0', port=port)