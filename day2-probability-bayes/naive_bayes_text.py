import numpy as np


p_spam=0.3
p_ham=0.7
p_words_spam={"free":0.8,"offer":0.7,"meeting":0.1}
p_words_ham={'free':0.1,'offer':0.05,'meeting':0.6}
def navie_bayes_predict(words:list[str])->str:
  log_spam=np.log(p_spam)
  log_ham=np.log(p_ham)
  for w in words:
    if w in p_words_spam:
      log_spam+=np.log(p_words_spam[w])
      log_ham+=np.log(p_words_ham[w])
  spam_score=np.exp(log_spam)
  ham_score=np.exp(log_ham)
  total=spam_score+ham_score
  return f'''P(spam|words):{spam_score/total},P(ham|words):{ham_score/total}'''
email=['free','offer']
print(f'''
Navie Bayes on email: {email}
{navie_bayes_predict(email)}
''')
