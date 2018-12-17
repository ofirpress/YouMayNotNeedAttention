# You May Not Need Attention

Code for the **Eager Translation Model** from the paper [You May Not Need Attention](https://arxiv.org/abs/1810.13409) by Ofir Press and Noah A. Smith. 

![Eager Translation Model](http://ofir.io/images/eagertranslation/eagertranslationmodel.png)

The following python packages are required:
* pytorch 0.4
* sacreBLEU
* mosestokenizer

In addition, [fast_align](https://github.com/clab/fast_align) is needed to compute alignments. 

This code requires Python 3.6+

## Preprocessing 
### Get the translation data
1. Download the dataset you'd like to use. For this example we'll use [Sockeye Autopilot](https://github.com/awslabs/sockeye/tree/master/sockeye_contrib/autopilot) to download the WMT 2014 EN->DE dataset.
```
sockeye-autopilot --task wmt14_en_de --model none
```
2. Enter the directory containing the bye-pair encoded (BPE) version of the data:
```
cd ./sockeye_autopilot/systems/wmt14_en_de/data/bpe
```
3. Unzip everything
```
gunzip *
```
4. Shuffle the tranining data
```
paste -d '|'  train.src train.trg | shuf | awk -v FS='|' '{ print $1 > "train.shuf.src" ; print $2 > "train.shuf.trg" }'
```


### Run the Eager Feasibility preprocessing

5. Combine the source and target training data into one file
```
paste -d ' ||| ' train.shuf.src - - - - train.shuf.trg < /dev/null > combined_srctrg
```

6. Use fast_align to find the aligments of the training sentence pairs
```
./fast_align -i ~/sockeye_autopilot/systems/wmt14_en_de/data/bpe/combined_srctrg -d -o -v > forward.align_ende
```

7. Run our script for making the training data Eager Feasible:
```
python  add_epsilons.py --align forward.align_ende --trg ~/sockeye_autopilot/systems/wmt14_en_de/data/bpe/to_train/train.shuf.trg --src ~/sockeye_autopilot/systems/wmt14_en_de/data/bpe/to_train/train.shuf.src --left_pad 4 --directory ~/corpus/WMTENDE/4pad/ 
```
Make sure the directory given for the --directory argument actually exists!
The --left_pad argument specifies how many initial padding tokens should be inserted into the training dataset. 

**Note:** The add_epsilons script may encounter sentence pairs which fast_align could not find an alignment for. If so, it will delete those lines from the training set. It will then ask you to re-run the script (with the same arguments) in order to finish the process. 



## Training
Use the following command to train:

```
python main.py --data ~/corpus/WMTENDE/4pad/  --save ~/exps/ --wdrop 0 --dropout 0.1 --dropouti 0.15 --dropouth 0.1 --dropoutcomb 0.1 --nlayer 4 --epochs 25 --bptt 60 --nhid 1000 --emsize 1000 --batch_size 200 --start_decaying_lr_step 200000 --update_interval 6500
```
These were the hyperparams used to train the models presented in the paper.

--save specifies where to store the model checkpoints

--nhid is the size of the LSTM 

--emsize is double the word embedding size, and must be equivalent to --nhid



## Translate
Once you have a trained model, you can use it to translate a document containing sentences in the source language.
```
python generate.py --checkpoint ~/exps/20181012-022002/model351000.pt  --data  ~/corpus/WMTENDE/4pad/    --src_path  ~/sockeye_autopilot/systems/wmt14_en_de/data/bpe/dev.src  --beam_size 5 --eval  --target_translation ~/sockeye_autopilot/systems/wmt14_en_de/data/raw/dev.trg  --epsilon_limit 3 --src_epsilon_injection 4  --start_pads 5 --language de --save_dir ./output/
```

This will translate the file found in --src_path , and will save the output in the --save_dir directory. In addition, it will compute the BLEU score. 


## Reference
If you found this code useful, please cite the following paper:

```
@article{press2018you,
  title={You May Not Need Attention},
  author={Press, Ofir and Smith, Noah A},
  journal={arXiv preprint arXiv:1810.13409},
  year={2018}
}
```

## Acknowledgments

This repository is based on the code from the [AWD-LSTM language model](https://github.com/salesforce/awd-lstm-lm). 
