#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
import _gdynet as dy
import numpy as np
import random
import sys
import time
import pickle
import json

dy.init()

UNK_THRESHOLD = 6
BATCH_SIZE = 32
LOG_FILE = 'large.log'
TRANSLATE_TEST_RESULT_FILE = 'large_test.result'
TRANSLATE_BLIND_RESULT_FILE = 'large_blind.result'
SAVE_MODEL_FILE = 'large.model'
MAX_LEN = 50

def read_file(fileName):
    resList = []
    f = open(fileName,'r')
    for line in f.xreadlines():
        # curr = ['<S>'] + line.replace('\n','').split(' ') + ['</S>']
        curr = ['<S>'] + line.replace('\n','').split(' ') + ['</S>']
        resList.append(curr)
    f.close()
    return resList

def create_batch(l):
    count = 0
    res = []
    while count < len(l):
        batch = [ins for ins in l[count:min([len(l),count+BATCH_SIZE])] if len(ins[0]) == len(l[count][0])]
        res.append(batch)
        count += len(batch)
    return res

class Attention():
    def __init__(self, model, training_src, training_tgt, dev_src, dev_tgt, test_src, blind_src, mode = 'train', modelFileName = '', dictFileName = ''):
        if mode == 'train':
            self.model = model
            self.training = [(x, y) for (x, y) in zip(training_src, training_tgt)]
            self.training.sort(key = lambda x: -len(x[0]))
            self.training_batch = create_batch(self.training)
            self.dev = [(x, y) for (x, y) in zip(dev_src, dev_tgt)]
            self.dev.sort(key = lambda x: -len(x[0]))
            self.dev_batch = create_batch(self.dev)
            self.test = test_src
            self.blind = blind_src
            self.src_token_to_id, self.src_id_to_token = self._buildMap(training_src)
            self.tgt_token_to_id, self.tgt_id_to_token = self._buildMap(training_tgt)
            self.src_vocab_size = len(self.src_token_to_id)
            self.tgt_vocab_size = len(self.tgt_token_to_id)
            self.embed_size = 512
            self.hidden_size = 512
            self.attention_size = 128
            self.layers = 1
            self.max_len = 50

            self.src_lookup = model.add_lookup_parameters((self.src_vocab_size, self.embed_size))
            self.tgt_lookup = model.add_lookup_parameters((self.tgt_vocab_size, self.embed_size))

            self.l2r_builder = dy.LSTMBuilder(self.layers, self.embed_size, self.hidden_size, model)
            self.l2r_builder.set_dropout(0.5)
            self.r2l_builder = dy.LSTMBuilder(self.layers, self.embed_size, self.hidden_size, model)
            self.r2l_builder.set_dropout(0.5)
            self.dec_builder = dy.LSTMBuilder(self.layers, self.embed_size + self.hidden_size * 2, self.hidden_size, model)
            self.dec_builder.set_dropout(0.5)
            self.W_y = model.add_parameters((self.tgt_vocab_size, self.hidden_size))
            self.b_y = model.add_parameters((self.tgt_vocab_size))

            self.W1_att_f = model.add_parameters((self.attention_size, self.hidden_size * 2))
            self.W1_att_e = model.add_parameters((self.attention_size, self.hidden_size)) 
            self.w2_att = model.add_parameters((self.attention_size))

        if mode == 'test':
            self.model = model
            self.load(modelFileName,dictFileName)
            self.test = test_src



    def save(self,modelFileName,dictFileName):
        self.model.save(modelFileName, [self.src_lookup,self.tgt_lookup,self.l2r_builder,self.r2l_builder,\
            self.dec_builder,self.W_y,self.b_y,self.W1_att_f,self.W1_att_e,self.w2_att])
        d = [self.src_token_to_id, self.src_id_to_token, self.tgt_token_to_id, self.tgt_id_to_token]
        pickle.dump(d, open(dictFileName,'wb'))


    def load(self,modelFileName,dictFileName):
        self.src_lookup,self.tgt_lookup,self.l2r_builder,self.r2l_builder,\
            self.dec_builder,self.W_y,self.b_y,self.W1_att_f,self.W1_att_e,self.w2_att = self.model.load(modelFileName)
        self.src_token_to_id, self.src_id_to_token, self.tgt_token_to_id, self.tgt_id_to_token = pickle.load(open(dictFileName,'rb'))


    def _buildMap(self,sents):
        token_to_id = {}
        id_to_token = {}
        token_to_id["<unk>"] = 0
        token_to_id["<S>"] = 1
        token_to_id["</S>"] = 2
        id_to_token[0] = '<unk>'
        id_to_token[1] = '<S>'
        id_to_token[2] = '</S>'
        currId = 3
        count = {}

        for s in sents:
            for token in s:
                count[token] = count.get(token,0) + 1
        for s in sents:
            for token in s:
                if token not in token_to_id and count[token] >= UNK_THRESHOLD:
                    token_to_id[token] = currId
                    id_to_token[currId] = token
                    currId += 1
        return (token_to_id, id_to_token)

    def _mlp(self, W1_att_f, W1_att_e, w2_att, h_fs_matrix, h_e, F):
        E = W1_att_e * h_e
        a = dy.colwise_add(W1_att_f * h_fs_matrix, E)
        res = dy.transpose(dy.tanh(a)) * w2_att
        return res

    # Calculates the context vector using a MLP
    # h_fs: matrix of embeddings for the source words
    # h_e: hidden state of the decoder
    def __attention_mlp(self, h_fs_matrix, h_e, F):
        W1_att_f = dy.parameter(self.W1_att_f)
        W1_att_e = dy.parameter(self.W1_att_e)
        w2_att = dy.parameter(self.w2_att)
        # Calculate the alignment score vector
        # Hint: Can we make this more efficient? 
        a_t = self._mlp(W1_att_f, W1_att_e, w2_att, h_fs_matrix, h_e, F) 
        alignment = dy.softmax(a_t)
        c_t = h_fs_matrix * alignment 
        return alignment, c_t

    # Training step over a single sentence pair
    def __step_batch(self, batch):
        dy.renew_cg()
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        F = len(batch[0][0])
        num_words = F * len(batch)

        src_batch = [x[0] for x in batch]
        tgt_batch = [x[1] for x in batch]
        src_rev_batch = [list(reversed(x)) for x in src_batch]
        # batch  = [ [a1,a2,a3,a4,a5], [b1,b2,b3,b4,b5], [c1,c2,c3,c4] ..]
        # transpose the batch into 
        #   src_cws: [[a1,b1,c1,..], [a2,b2,c2,..], .. [a5,b5,</S>]]

        src_cws = map(list, zip(*src_batch)) # transpose
        src_rev_cws = map(list,zip(*src_rev_batch))

        # Bidirectional representations
        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()
        l2r_contexts = []
        r2l_contexts = []
        for (cw_l2r_list, cw_r2l_list) in zip(src_cws,src_rev_cws):
            l2r_state = l2r_state.add_input(dy.lookup_batch(self.src_lookup, [self.src_token_to_id.get(cw_l2r,0) for cw_l2r in cw_l2r_list]))
            r2l_state = r2l_state.add_input(dy.lookup_batch(self.src_lookup, [self.src_token_to_id.get(cw_r2l,0) for cw_r2l in cw_r2l_list]))
            
            l2r_contexts.append(l2r_state.output()) #[<S>, x_1, x_2, ..., </S>]
            r2l_contexts.append(r2l_state.output()) #[</S> x_n, x_{n-1}, ... <S>]

        r2l_contexts.reverse() #[<S>, x_1, x_2, ..., </S>]

        # Combine the left and right representations for every word
        h_fs = []
        for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
            h_fs.append(dy.concatenate([l2r_i, r2l_i]))
        h_fs_matrix = dy.concatenate_cols(h_fs)

        losses = []
        

        # Decoder
        # batch  = [ [a1,a2,a3,a4,a5], [b1,b2,b3,b4,b5], [c1,c2,c3,c4] ..]
        # transpose the batch into 
        #   tgt_cws: [[a1,b1,c1,..], [a2,b2,c2,..], .. [a5,b5,</S>]]
        #   masks: [1,1,1,..], [1,1,1,..], ...[1,1,0,..]]

        tgt_cws = []
        masks = []
        maxLen = max([len(l) for l in tgt_batch])
        for i in range(maxLen):
            tgt_cws.append([])
            masks.append([])
        for sentence in tgt_batch:
            for j in range(maxLen):
                if j > len(sentence) - 1:
                    tgt_cws[j].append('</S>')
                    masks[j].append(0)
                else:
                    tgt_cws[j].append(sentence[j])
                    masks[j].append(1)
        c_t = dy.vecInput(self.hidden_size * 2)
        start = dy.concatenate([dy.lookup(self.tgt_lookup, self.tgt_token_to_id['<S>']), c_t])
        dec_state = self.dec_builder.initial_state().add_input(start)
        for (cws, nws, mask) in zip(tgt_cws, tgt_cws[1:], masks):
            h_e = dec_state.output()
            _, c_t = self.__attention_mlp(h_fs_matrix, h_e, F)
            # Get the embedding for the current target word
            embed_t = dy.lookup_batch(self.tgt_lookup, [self.tgt_token_to_id.get(cw,0) for cw in cws])
            # Create input vector to the decoder
            x_t = dy.concatenate([embed_t, c_t])
            dec_state = dec_state.add_input(x_t)
            y_star = dy.affine_transform([b_y, W_y, dec_state.output()])
            loss = dy.pickneglogsoftmax_batch(y_star, [self.tgt_token_to_id.get(nw,0) for nw in nws])
            mask_exp = dy.reshape(dy.inputVector(mask),(1,),len(mask))
            loss = loss * mask_exp
            losses.append(loss)
        return dy.sum_batches(dy.esum(losses)) , num_words

    def train(self):
        trainer = dy.SimpleSGDTrainer(self.model)
        # trainer = dy.AdamTrainer(self.model)
        step = 0
        logFile = open(LOG_FILE,'w+')
        for epoch in range(1000):
            count = 0
            step = 0
            random.shuffle(self.training_batch)
            lastTime = time.time()
            for batch in self.training_batch:
                count += len(batch)
                loss, num_words = self.__step_batch(batch)
                # train_loss += loss.npvalue()[0]
                loss.backward()
                trainer.update()

                if count >= step * 5000:
                    logFile.write('last 5000 samples time: ' + str(time.time() - lastTime) + '\n')
                    lastTime = time.time()
                    step += 1
                    dev_loss = 0
                    for batch_dev in self.dev_batch:
                        loss, num_words = self.__step_batch(batch_dev)
                        dev_loss += loss.npvalue()[0]
                    
                    logFile.write('epoch: ' + str(epoch) + '\n')
                    logFile.write('dev_loss: ' + str(dev_loss) + '\n')
                    # logFile.write('train_loss: ' + str(train_loss) + '\n')
                    logFile.flush()
                    # train_loss = 0

            f = open(TRANSLATE_TEST_RESULT_FILE,'w')
            print 'Writing current test result...'
            for sent in self.test:
                f.write(self.translate_sentence(sent))
                f.write('\n')
            f.close()

            f = open(TRANSLATE_BLIND_RESULT_FILE,'w')
            print 'Writing current blind result...'
            for sent in self.blind:
                f.write(self.translate_sentence(sent))
                f.write('\n')
            f.close()

            print 'Result written.'
            self.save(SAVE_MODEL_FILE)

    def translate_sentence(self, sent):
        dy.renew_cg()
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        F = len(sent)

        sent_rev = list(reversed(sent))

        # Bidirectional representations
        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()
        l2r_contexts = []
        r2l_contexts = []
        for (cw_l2r, cw_r2l) in zip(sent, sent_rev):
            l2r_state = l2r_state.add_input(dy.lookup(self.src_lookup, self.src_token_to_id.get(cw_l2r,0)))
            r2l_state = r2l_state.add_input(dy.lookup(self.src_lookup, self.src_token_to_id.get(cw_r2l,0)))
            l2r_contexts.append(l2r_state.output()) #[<S>, x_1, x_2, ..., </S>]
            r2l_contexts.append(r2l_state.output()) #[</S> x_n, x_{n-1}, ... <S>]
        r2l_contexts.reverse()

        h_fs = []
        for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
            h_fs.append(dy.concatenate([l2r_i, r2l_i]))
        h_fs_matrix = dy.concatenate_cols(h_fs)

        # Decoder
        trans_sentence = ['<S>']
        cw = trans_sentence[-1]
        c_t = dy.vecInput(l2r_contexts[0].npvalue().shape[0] * 2)
        start = dy.concatenate([dy.lookup(self.tgt_lookup, self.tgt_token_to_id['<S>']), c_t])
        dec_state = self.dec_builder.initial_state().add_input(start)
        while len(trans_sentence) < MAX_LEN:
            h_e = dec_state.output()
            alignment, c_t = self.__attention_mlp(h_fs_matrix, h_e, F)
            embed_t = dy.lookup(self.tgt_lookup, self.tgt_token_to_id.get(cw,0))
            # Create input vector to the decoder
            x_t = dy.concatenate([embed_t, c_t])
            dec_state = dec_state.add_input(x_t)
            y_star = dy.affine_transform([b_y, W_y, dec_state.output()])
            cw = self.tgt_id_to_token[np.argmax(y_star.npvalue())]
            if cw == '<unk>': #unknown words replacement
                a = alignment.npvalue()
                if np.argmax(a) == 0:
                    a[0] = 0
                if np.argmax(a) == len(a) - 1:
                    a[len(a) - 1] = 0
                trans_sentence.append(sent[np.argmax(a)])
                continue
            if cw == '</S>':
                break
            trans_sentence.append(cw)

        return ' '.join(trans_sentence[1:])

    def translate_sentence_beam_compact(self, sent, k = 2):
        dy.renew_cg()
        F = len(sent)
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        sent_rev = list(reversed(sent))
        # Bidirectional representations
        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()
        l2r_contexts = []
        r2l_contexts = []
        for (cw_l2r, cw_r2l) in zip(sent, sent_rev):
            l2r_state = l2r_state.add_input(dy.lookup(self.src_lookup, self.src_token_to_id.get(cw_l2r,0)))
            r2l_state = r2l_state.add_input(dy.lookup(self.src_lookup, self.src_token_to_id.get(cw_r2l,0)))
            l2r_contexts.append(l2r_state.output()) #[<S>, x_1, x_2, ..., </S>]
            r2l_contexts.append(r2l_state.output()) #[</S> x_n, x_{n-1}, ... <S>]
        r2l_contexts.reverse()

        h_fs = []
        for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
            h_fs.append(dy.concatenate([l2r_i, r2l_i]))
        h_fs_matrix = dy.concatenate_cols(h_fs)
        valid = [True for i in range(k)]
        trans = [['<S>'] for i in range(k)]
        prob = [0 for i in range(k)]
        used = [set([0, F-1]) for i in range(k)]
        cw = [trans[i][-1] for i in range(k)]
        c_t = dy.vecInput(l2r_contexts[0].npvalue().shape[0] * 2)
        start = dy.concatenate([dy.lookup(self.tgt_lookup, self.tgt_token_to_id['<S>']), c_t])
        dec_state = [self.dec_builder.initial_state().add_input(start) for i in range(k)]
        b = [i for i in range(k)]
        FIRST = True
        while True in valid: 

            h_e = [dec_state[i].output() for i in range(k)]
            ac = [self.__attention_mlp(h_fs_matrix, elem, F) for elem in h_e]
            alignment = [elem[0] for elem in ac]
            c_t = [elem[1] for elem in ac]
            embed_t = [dy.lookup(self.tgt_lookup, self.tgt_token_to_id.get(cw[i],0)) for i in range(k)]
            x_t = [dy.concatenate([embed_t[i], c_t[i]]) for i in range(k)]
            dec_state = [dec_state[b[i]].add_input(x_t[i]) for i in range(k)]
            y_star =[dy.affine_transform([b_y, W_y, dec_state[i].output()]) for i in range(k)]
            p = [dy.log_softmax(y_star[i]) for i in range(k)]
            tmp = [p[i].npvalue() for i in range(k)]
            l = []
            if not FIRST:
                idx = [np.argpartition(-tmp[i].T,k)[0][:k] for i in range(k)]
                val = [-np.partition(-tmp[i].T,k)[0][:k] for i in range(k)]
                for i in range(k):
                    for j in range(k):
                        l.append((self.tgt_id_to_token[idx[i][j]],val[i][j] + prob[i],i))

            else:
                idx = np.argpartition(-tmp[0].T,k)[0][:k]
                val = -np.partition(-tmp[0].T,k)[0][:k]
                FIRST = False
                for i in range(k):
                    l.append((self.tgt_id_to_token[idx[i]],val[i],i))

            l = sorted(l, key = lambda x:-x[1])[:k]
            # print l
            cw = [l[i][0] for i in range(k)]
            prob = [l[i][1] for i in range(k)]
            b = [l[i][2] for i in range(k)]
            
            trans = [list(trans[b[i]]) for i in range(k)]
            valid = [valid[b[i]] for i in range(k)]

            for i in range(k):
                if valid[i] == False: continue
                if cw[i] != '</S>' and len(trans[i]) < MAX_LEN:
                    if cw[i] == '<unk>': #unknown words replacement
                        a = alignment[i].npvalue()
                        am = np.argmax(a)
                        while am in used:
                            a[am] = -1
                            am = np.argmax(a)
                        used[i].add(am)
                        if len(used[i]) == F / 2:
                            used[i] = set([0, F-1])
                        trans[i].append(sent[am])
                    else:
                        trans[i].append(cw[i])
                else:
                    valid[i] = False
                
        index = prob.index(max(prob))
        return ' '.join(trans[index][1:])


def main():
    model = dy.Model()
    trainer = dy.SimpleSGDTrainer(model)
    if sys.argv[1] == '-train':
        training_src = read_file(sys.argv[2])
        training_tgt = read_file(sys.argv[3])
        dev_src = read_file(sys.argv[4])
        dev_tgt = read_file(sys.argv[5])
        test_src = read_file(sys.argv[6])
        blind_src = read_file(sys.argv[7])
        attention = Attention(model, training_src, training_tgt, dev_src, dev_tgt, test_src, blind_src)
        attention.train()
        

    elif sys.argv[1] == '-trans':
        test_src = read_file(sys.argv[2])
        attention = Attention(model, None, None, None, None, test_src, None, 'test')


if __name__ == '__main__': main()
