We received great help from https://github.com/akarshzingade/image-similarity-deep-ranking.
Special thanks to Akarsh Zingade for helping us about Visnet. 

We added test.py to show the ranking of the images. 

This model is based on the paper, https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf. For detailed information and explanation, please refer to the paper and the blog https://medium.com/@akarshzingade/image-similarity-using-deep-ranking-c1bd83855978. 

How to run this code.

1. You need image folders containing the data images, each folder represents a class. and run  
python triplet_sampler.py --input_directory <<path to the directory>> --output_directory <<path to the directory>> --num_pos_images     <<Number of positive images you want>> --num_neg_images <<Number of negative images you want>>

This can create a triplet.txt, each line contains the query image, positive and negative image absolute directory path. 

2. Change the directory path in train.py and test.py to fit your dataset. 

3. Run train.py.

4. Run test.py.
