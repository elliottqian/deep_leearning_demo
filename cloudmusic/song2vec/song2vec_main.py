# -*- coding: utf-8 -*-

import codecs
import os
import random

import gensim
import gensim.models
import sys

sys.path.append("/mnt/D/Ubuntu/package/anaconda3/lib")

class MySentences(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name

    def __iter__(self):
        for f_name in os.listdir(self.dir_name):
            for line in codecs.open(os.path.join(self.dir_name, f_name)):
                l = line.split("\t")[-1].split(",")
                r = list(map(lambda x: x.split(":")[0], l))
                yield r


def main(path, model_path):
    sentences = MySentences(path)
    model = gensim.models.Word2Vec(sentences, size=50, iter=20, window=5, min_count=0, workers=8)
    model.wv.vocab["493042772"]
    model.save(model_path)


def load_model(model_name):
    model = gensim.models.Word2Vec.load(model_name)
    print(model.most_similar("255526"))
    print(model.wv.syn0norm.shape)
    print(model.wv.vocab["493042772"].index)       #  word to index
    print(model.wv.syn0[62])
    print(model["255294"])
    # for key in model.wv.vocab:
    #     print(key)
    #     print(model.wv.vocab[key])
    return model

if __name__ == "__main__":
    path_ = "/media/elliottqian/专业资料/数据/网易云音乐用户听歌数据"
    model_path_ = "/home/elliottqian/Documents/PycharmProjects/deep_leearning_demo/cloudmusic/song2vec/model.song2vec"
    # sentences = MySentences(path_)
    # print(next(sentences.__iter__()))
    # main(path_, model_path_)
    model = load_model(model_path_)
    # [('29431066', 0.9836549758911133), ('30431376', 0.9821330308914185), ('415086030', 0.9748191833496094), ('32957955', 0.9735628366470337), ('420400437', 0.9701138734817505), ('38576323', 0.9697043895721436), ('484732973', 0.969683051109314), ('34923114', 0.9680273532867432), ('25706285', 0.9678186178207397), ('29535690', 0.9637477993965149)]

    # print(model.get_latest_training_loss())
    pass
