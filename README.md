## LT2316 H20 Assignment A2 : Ner Classification

Name: Merle Pfau

## Notes on Part 1.

I chose to build a model with n LSTM layers for my NER classifier. Batches of data are put through an LSTM with n layers and then classified in a linear layer by taken the last hidden state at converting it into labels. The input dimensions of the model are the number of features in the data, the output the number of words in a sentence (max_sample_len). I included a Dropout layer in the LSTM to prevent overfitting and introduce additional randomness in my model.
I chose an LSTM since I thought the memory in using the last hidden state in the model would help in the classifying process. I read that classic RNNs could be unable to work with longer sequences and hold on to long-term dependencies, making them suffer from “short-term memory”.

## Notes on Part 2.

For load_model() I chose to return epoch, model_state_dict, optimizer_state_dict, hyperparamaters, loss, scores and model_name, so that I would have everything needed for the testing. My training loop consists of batching the data (I used the Batcher from the Cats & Horses Demos), putting the feature vectors of one batch through the model and calculating the mean absolute error.
After a model was trained I evaluated it with the val split, getting accuracy, recall, precision and f1 score.
The testing is similar to the evaluation, I load a model and get the same scores as above.

## Notes on Part 3.

The parameters I chose were: "learning_rate", "hidden_size", "number_layers", "optimizer" and "batch_size". I tested some with the Adam optimizer and some with the SGD but couldn't see one outperforming the other.
The parallel coordinates plot was not that helpful to me, as it got a bit messy and hard to read, so I relied more on my parameter definitions as well as the printing of all scores. I still relied on the same metric that I passed to it, the F1 score, since it combines some of the other scores I calculated.
After the first five models were trained, I took the three best performing ones and slightly changed some variables. 

## Notes on Part 4.

All of my models seemed to get stuck in the training with the loss not significantly decreasing after the first few epochs. The f1 score when testing on the val set was still not bad, with mdoel 1 reaching 90,5%. I therefore chose to use that model for testing, but only reached a score of about 5%. Something seems to be going wrong, I have not figured out what it is though.

## Notes on Part Bonus.

*fill in notes and documentation for the bonus as mentioned in the assignment description, if you choose to do the bonus*
