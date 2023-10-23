# kaggle CommonLit - Evaluate Student Summaries - 4th solution training code (chumajin part) 
This repository is designed to facilitate the execution of training code for the Kaggle CommonLit - Evaluate Student Summaries.

# 0. Environment

I created the models using Google Colab Pro+. I used an A100 GPU with high memory.

In that case, at least you must install adder libraries as follows:
~~~
!pip install transformers==4.33.1
!pip install sentencepiece==0.1.99
~~~

In other cases, please refer to the requirement.txt file for further instructions.

* python : 3.10.12
* cuda : 12.0
* driver version 525.105.17


# 1. DataPreparation

The input data is required to have the following structure (same in the kaggle competition dataset).

~~~
inputpath
    ├── prompts_text.csv
    ├── prompts_train.csv
    ├── sample_submission.csv
    ├── summaries_text.csv
    └── summaries_train.csv
~~~

# 2. Select model

The following training code makes only one model weight. Please specify the **model no** you would like to use.

| model no | InfNo | model                   | training<br/>maxlen | inference maxlen | input           | pooling |           |      | 2nd loss | preprocess | cv of 4kfold earlystop |
|---------|-------|-------------------------|---------------------|------------------|-----------------|---------|-----------|------|----------|------------|------------------------|
|         |       |                         |                     |                  | original prompt | cls     | attention | mean |          |            |                        |
| 2       | 22    | deberta-v3-large        | 1050                | 950              | ✓               |         | ✓         |      |          |            | 0.4855                 |
| 3       | 63    | deberta-v3-large        | 850                 | 1500             |                 |         |           | ✓    |          |            | 0.4984                 |
| 4       | 72    | deberta-v3-large        | 868                 | 1024             |                 | ✓       |           |      | Arcface  | ✓          | 0.4919                 |
| 5-1     | 2     | deberta-v3-large        | 868                 | 868              |                 | ✓       |           |      |          |            | 0.4880                 |
| 5-2     | 3     | deberta-v3-large        | 868                 | 868              |                 | ✓       |           |      |          |            | 0.4979                 |
| 7       | 331   | deberta-v3-large-squad2 | 1050                | 950              | ✓               |         | ✓         |      |          |            | 0.4993                 |

※ model no 1 and model no 6 can make by using kuro_B's training code


# 3. Execute training

Run the following code. The model weight will be created inside the savepath as 'model{model no}.pth'. If you provide a folder name as savepath, a folder will be created automatically.  


~~~
python train.py \
--modelno {model no} \
--train_fold 4 \
--savepath {savepath} \
--inputpath {inputpath}
~~~

example :
In this case, I explain the case of making model no 3 in the table of chapter 2. The savepath (./output folder) will make and the model weight will save in it as model3.pth.

~~~
python train.py \
--modelno 3 \
--train_fold 4 \
--savepath output \
--inputpath $inputpath
~~~

In this case, train_fold 4 means the fulltrain model used at the end of the competition. If you want to create it with 4kfold, change it like this : 0 1 2 3. Also, if you are using a different environment, and GPU memory is not enough, please adjust train_batch, valid_batch in the py file per model no in config. In that case, it is recommended that you also change layerwise_lr to give the condition.

# 4. inference

For kaggle competition, you can use the inference code on the kaggle notebook [here](https://www.kaggle.com/code/chumajin/commonlit2-4th-place-inference). After you upload your model to kaggle dataset, you replace the modelpath in each according to the model no and Infno(EXP).
The following code is the example that replaces the model3.pth.

In the kaggle code

~~~
# 3.modelno3 : expno 63
~~~

~~~
!python exp224fs_maxlen1500wopp.py --EXP 63 \
    --modelpath /kaggle/input/commonlit2-4th-place-models/model4_seed237.pth \
  #  --debug True
~~~

Example for inference using model3.pth you made.
~~~
!python exp224fs_maxlen1500wopp.py --EXP 63 \
    --modelpath /kaggle/input/****/model3.pth \
  #  --debug True
~~~

※ For **** in the modelpath, you must replace to your dataset path in the kaggle dataset.

# 5. license

The code in this repository is MIT lisence.






