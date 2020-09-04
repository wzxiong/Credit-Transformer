# Credit-Transformer
使用Tensorflow2.0训练nlp模型
textClassiferModel.py 对应nlp_transformer.py和config.ini.nlp，输入的数据是所有nlp分词后的index,结构为纯Transformer

数据样例： id,date \t	label \t	43105,40087,63036

textClassiferModelMulti.py 对应all_transformer.py和config.ini.all2，输入的数据是连续特征+时序特征+nlp分词后的index，结构为FNN + Tran + Tran
textClassiferModelMultiReshape.py 对应allreshape_transformer.py和config.ini.all3，输入的数据是连续特征+时序特征+nlp分词后的index，结构为FNN + Tran + Tran
