# ML sequence to sequence modeling

This script implements a basic character-level sequence-to-sequence model to translate human readable dates ("25th of June, 2009") into "machine" uniquely-formatted readable dates ("2009-06-25").

## Motivation
The use case is largely motivated by a programming assignemnt on the Deep Learning Specialisation on Coursera (by Andrew Ng) in which an attention model is used. I asked myself how I could tackle the same problem with an encoder-decoder architecture leveraging LSTMs (which I haven't found anywhere). Since my second motivation was to test the Keras library, I took the example given here (https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) as a starting point and adapted it to this specific problem.
Of course, the main motivation was to figure out how to implement these architectures and to go beyond a theoretical understanding.

#### The basic steps of the algorithm are:
##### Training

- Start with input sequences (human readable dates)
    and the corresponding target sequences (dates in a standard format).
- An encoder LSTM turns input sequences into two state vectors
    (we keep the last LSTM states and discard the outputs).
- A decoder LSTM is trained to predict at each time step the next character in the target sequence (a training process called "teacher forcing" in this context). The decoder uses as initial state the state vectors returned by the encoder. Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
    
##### Inference
In inference mode, when we want to decode unknown input sequences, we:
- Encode the input sequence into state vectors.
- Start with a target sequence of size 1 (just the start-of-sequence character)
- Feed the state vectors and 1-char target sequence
to the decoder to produce predictions for the next character
(in fact, a probability distribution for the next char is generated)
- Sample the next character using these predictions (by simply using argmax).
- Append the sampled character to the target sequence.
- Repeat until we generate the full date (10 characters). In this case the length of the output is fixed. In a transalation problem from one language to another, we should repeat the process until we generate the end-of-sentence character.  

# Data
Dates are provided in ``dataset.csv``

# References
- Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215
- Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    https://arxiv.org/abs/1406.1078'''