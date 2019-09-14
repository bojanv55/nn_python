import numpy as np

def softmax(z):
   return np.array([(np.exp(el)/np.sum(np.exp(z))) for el in z])

def cost(W,F,L):
   m = F.shape[0] #get number of rows
   mul = np.dot(F, W)
   sm_mul_T = softmax(mul)
   return -(1/m) * np.sum(L * np.log(sm_mul_T))

def gradient(W,F,L):
   m = F.shape[0]  # get number of rows
   mul = np.dot(F, W)
   sm_mul_T = softmax(mul)
   return -(1 / m) * np.dot(F.T , (L - sm_mul_T))

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./datasets/MNIST_data/", one_hot=True)

W = np.zeros((785, 10)) #784 features + 1 bias

for _ in range(1000):
   F, L = mnist.train.next_batch(100)

   F = np.insert(F,0, values=1, axis=1)

   total_cost = cost(W,F,L)
   print("Total cost is {}".format(total_cost))

   gradients = gradient(W,F,L)

   W = W - (0.001 * gradients)

FU = mnist.test.images

FU = np.insert(FU,0, values=1, axis=1)

LU = mnist.test.labels

mulU = np.dot(FU, W)
sm_mulU = softmax(mulU)

OK=0
NOK=0

for i in range(10000):
   a1 = np.argmax(sm_mulU[i])
   a2 = np.argmax(LU[i])
   if a1 == a2:
      OK = OK + 1
   else:
      NOK = NOK + 1

print("{} OK vs {} NOK".format(OK, NOK))
print("accur {}%".format((OK/(NOK+OK))*100))
