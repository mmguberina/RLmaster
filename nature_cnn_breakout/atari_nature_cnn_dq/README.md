# Nature CNN
This code is based on this research paper
 https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf

Additional modules required
numpy : 1.11.0 
pythorch : virutual env(pip install gym torch)
gym[atari] : pip install 'msgpack== 1.0.2' gym['atari']
tensorboard : pip install tensorbpard


Files:
-baseline_wrappers: taken from openai/baselines
-pytorch wrappers:
-main.py:
-observe.py:
-play.py:
-msgpack.py: copied from github repo taken from openai/baselines,used to serails numpy arraies uisng msgpack
-logs: a log file that contiains information on number of average episodes and average reward
-atari_model_LR.pack : the trained model to saved for later accessblity,like see the ai playing


Summary Preprocessing steps:

- Image reascale and GrayScale(210*160 ->  84*84) and Frame stacking (k = 4) :the output image of the enviroment is converted to gray scale and rescaled to 84*84 image that
will reduce the size from 210*160,and each of the last four frames the agent has seen are restacked into a single observation this allows the agent to take information from each of those frames when its making a descision
-Reward clipping [-1,1]:In the paper all postive rewards are at 1 and all negative rewars at -1 are clipped.
-
-Frame skipping:allowing the agent only to interact with only every fourth frame of the eviroment and repeats it actions on the last frame.
-No-ops actions at the beiging of each episode (max = 30),do nothing action at the beinging of each episode.This adds stocaticity and randomness  in the enviromeant .

-the learning rate is used as a suffix on the saved model to indentifiy the model version

The general tasks completed in this version
1: set up enviroment and add starter codes
2: ADD conv_net 
3: save and load the model 
4: support GPU traing
5: train atrai breakout





ps:this file is under working progress the documentaion is temporar



