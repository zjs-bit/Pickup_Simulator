
import tensorflow as tf

z1 = tf.Variable(2.0)
z2 = tf.Variable(2.0)
z1marg = -5
z1margadj=z1marg+10

def log_pred_prob(z1,z2,outcome,gameto):

    outfrac = outcome/gameto*2
    return gameto*tf.nn.sigmoid_cross_entropy_with_logits(labels=[outfrac],logits=[z1-z2])

with tf.GradientTape() as tape:
    logpred = log_pred_prob(z1,z2,z1margadj,gameto=10)

print(logpred.numpy())
print(tape.gradient(logpred,[z1,z2]))