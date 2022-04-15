The high level goal at the moment can be summarized as:
"become a reinforcement learning practitioner" which roughly constists of 
4 points:
1. be able to turn equations and concepts into code
2. be able to debug implementations in a reasonable amount of time
3. get a feel for where different algorithms perform well, 
guesstimate reasonable hyperparameters and how hard different problems are
4. combine different codes into new augmented algorithms with speed and accuracy

Since these points are abstract and I need to perform concrete tasks,
an outcome which represents these outcomes needs to be defined.
It should be clearly defined, contrained in scope, related to the research topic
and not too difficult.
Taking everything into account, the goal is:
"get perfect performance on flappy bird from pixels with a detailed background",
or in other words completely solve flappy bird on the highest difficulty.
Once this is done, try to do the same with highest possible sample efficiency.
This functions the same way as Atari games, but it is easier to control 
settings-wise as I already know how to do just states, have a single colored background etc.
Thus the progressions should be natural.
Of course, because it is simple it will be a nice way to test different algorithms.
Furthermore, I anticipate that the most difficult part will be to extract the state 
from the image observations and that's what we're trying to deal with research-wise.
Finally, there's an implementation with model predictive control which performs really well
which means that it is possible to gather samples to try immitation learning as well.
Once I get more into it, I'll elaborate different stages of the problem and 
track progress in great detail.
