
#download glove
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d glove
rm glove.6B.zip

#download the image feature
wget -P coco https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip
unzip coco/trainval_36.zip -d coco/
rm coco/trainval_36.zip

#download vqacp2
wget -P vqacp2/ https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_questions.json
wget -P vqacp2/ https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_questions.json
wget -P vqacp2/ https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_annotations.json
wget -P vqacp2/ https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_annotations.json

