# Inverse-Design of Neuromorphic Computing Hardware

This study looks at the problem of limited memory capacity in nanomagnetic networks,
which makes it harder for them to perform tasks like classification and prediction that need
memory, similar to what recurrent neural networks (RNNs) do. To solve this problem, we
inversely designed a delay-line system by adding magnetisation masks and changing the loss
function using machine learning techniques, allowing it to store and use past information better.
We tested the system by doing RF signal classification and pattern recognition tasks, where it
achieved 99% accuracy in RF signal classification and 87.3% in pattern recognition, show a
clear improvement in handling tasks in time domain. We also measured the system’s memory
capacity and found that these changes not only improved its performance in tasks but also
helped it remember and use past inputs better. In conclusion, the changes to the delay-line
system greatly improved its memory capacity, bringing it closer to the abilities of RNNs for
time-domain tasks. These improvements show that nanomagnetic networks could be used more
widely in neuromorphic computing, offering a way to lower energy use in machine learning
systems while making them more efficient.

<img width="409" alt="image" src="https://github.com/user-attachments/assets/ea3508e0-bbb2-4532-8227-2da91ba693c5">

inverse design dealy-line structure

<img width="423" alt="image" src="https://github.com/user-attachments/assets/46d49174-bea2-4f1a-a11b-7cceff7f24a1">

confusion matrix of the pattern recognition, with accuracy 87%



