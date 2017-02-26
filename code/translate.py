from collections import defaultdict
import _gdynet as dy
import numpy as np
import random
import sys
import time

from attention_minibatch import Attention
from attention_minibatch import read_file
dy.init()

saved_model = sys.argv[1]
saved_dict = sys.argv[2]
test_src = read_file(sys.argv[3])

model = dy.Model()
attention = Attention(model, None, None, None, None, test_src, None, 'test', saved_model, saved_dict)
print 'model loaded'
for bw in range(1,10):
    f = open('test_output_beam_'+str(bw),'w')
    for sent in attention.test:
        f.write(attention.translate_sentence_beam_compact(sent,bw))
        f.write('\n')
        f.flush()
    f.close()





