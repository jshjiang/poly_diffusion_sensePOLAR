from scripts.SensePOLAR import bertFuncs as func
import numpy as np
import torch

def SensePolar(tokenizer, model, context, word, antonym_path = "", \
                                    normalize_term_path="scripts/SensePOLAR/antonyms/wordnet_normalize.pkl", \
                                    verbose=False):
    # forward the word
    cur_word_emb = func.forwardWord(tokenizer, model, context, word)
    # Normalization
    if normalize_term_path is not None:
        import pickle
        with open(normalize_term_path, 'rb') as curAntonymsPickle:
            normalize_term = pickle.load(curAntonymsPickle)
        cur_word_emb = cur_word_emb - normalize_term
    #get polar space
    if antonym_path =="":
        antonym_path = "scripts/SensePOLAR/antonyms/antonym_wordnet_base_change.pkl"
    _, W_inv_np = func.getW(antonym_path)
    W_inv_torch = torch.from_numpy(W_inv_np)

    #base-change into polar space
    polar_emb = torch.matmul(W_inv_torch,cur_word_emb)
    word_embedding=polar_emb.numpy()


    thisdict = {}
    for count, value in enumerate(word_embedding):
        thisdict[count] = value

    subspace = np.array(list(thisdict.values()))

    if verbose:
        antonym_path="scripts/SensePOLAR/antonyms/lookup_anto_example_dict.pkl"
        definition_path="scripts/SensePOLAR/antonyms/lookup_synset_definition.pkl"

        # For matchind dimension to antonym pair
        with open(antonym_path, 'rb') as curAntonymsPickle:
            antonyms = pickle.load(curAntonymsPickle)
        # for retrieving wordnet definitions of the antonym pair
        with open(definition_path, 'rb') as curDefPickle:
            definitions = pickle.load(curDefPickle)

        # sort the embedding on absolute value
        sortedDic = sorted(thisdict.items(), key=lambda item: abs(item[1]))
        sortedDic.reverse()

        axis_list=[]
        # Retrieve and print top-numberPolar dimensions
        for i in range(0, 5):
            cur_Index = sortedDic[i][0]
            cur_value = sortedDic[i][1]

            leftPolar = antonyms[cur_Index][0][0]
            leftDefinition = definitions[cur_Index][0]

            rightPolar = antonyms[cur_Index][1][0]
            rightDefinition = definitions[cur_Index][1]

            axis=leftPolar + "---" + rightPolar
            axis_list.append(axis)

            # Print
            print("Top: ", i)
            print("Dimension: ", leftPolar + "<------>"+ rightPolar)
            print("Definitions: ", leftDefinition+ "<------>"+ rightDefinition)
            if cur_value <0:
                print("Value: " + str(cur_value))
            else:
                print("Value:                      " + str(cur_value))
            print("\n")

    return subspace
