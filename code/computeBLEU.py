import nltk
from nltk.compat import Fraction
import sys
def method2(p_n, *args, **kwargs):
        """
        Smoothing method 2: Add 1 to both numerator and denominator from
        Chin-Yew Lin and Franz Josef Och (2004) Automatic evaluation of
        machine translation quality using longest common subsequence and
        skip-bigram statistics. In ACL04.
        """
        return [Fraction(p_i.numerator + 1, p_i.denominator + 1, _normalize=False) for p_i in p_n]

def corpusBLEU(resultFile, refFile):
    result = []
    ref = []
    bleu = 0
    f = open(resultFile,'r')
    for line in f.xreadlines():
        result.append(line.replace('\n','').split(' '))
    f.close()

    f = open(refFile, 'r')
    for line in f.xreadlines():
        ref.append([line.replace('\n','').split(' ')])
    f.close()
    return nltk.translate.bleu_score.corpus_bleu(ref,result)

if __name__ == '__main__':
    # hypo = sys.argv[1]
    # ref = sys.argv[2]
    ref = '../en-de/test.en-de.low.en'
    for hypo in range(1,7):
        print corpusBLEU('test_output_beam_' + str(hypo),ref)
