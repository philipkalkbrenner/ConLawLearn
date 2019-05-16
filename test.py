import tensorflow as tf
from splitting_stress import EffectiveStressSplit

A = tf.constant([[-500085,-763109,-2.0017e+06]])

Princ = EffectiveStressSplit.GetPrincipalDirection(A)

Pos = EffectiveStressSplit.GetPositiveStress(A)
Neg = EffectiveStressSplit.GetNegativeStress(A)

with tf.Session() as sess:
    print("A", sess.run(A))
    print("PRINC", sess.run(Princ))
    print("POS",sess.run(Pos))
    print("NEG",sess.run(Neg))
