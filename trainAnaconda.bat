cd D:;
cd D:\Projects\code2vec;
conda activate code2vec;
python code2vec.py --data data/java-small/java-small --test data/java-small/java-small.val.c2v --save models/java-small/saved_model;
pause

cd D:;
cd D:/Projects/code2vec;
conda activate code2vec;
python code2vec.py --data data/java-small/java-small --test data/java-small/java-small.val.c2v --save models/java-small/saved_model


#export vectors and interact...?
python code2vec.py -l models/java-small/saved_model --predict --export_code_vectors --data data/java-small/java-small -t2v data/java-small/target-embeddings -w2v data/java-small/token-embeddings

#import existing large model
python code2vec.py -v 2 -l data/java-large-release/java-large-released-model.tar --predict --export_code_vectors -t2v data/java-large-release/target-embeddings -w2v data/java-large-release/token-embeddings

#run interactive on existing large model
python code2vec.py  -v 2 --load models/java-large-release/saved_model_iter3.release --predict --export_code_vectors 