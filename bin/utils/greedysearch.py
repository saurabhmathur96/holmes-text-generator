import numpy as np


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def greedysearch(sentence, predict, model, maxsample):
    sentences = []
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        generated =  sentence
        
        for i in range(maxsample):
            preds = predict(model, generated)
            next_index = sample(preds, diversity)
            generated.append(next_index)
        sentences.append(generated)
    return sentences