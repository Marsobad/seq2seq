import torch


def beam_search(beam_size, decoder, initial_decoder_input, max_length, initial_decoder_hidden, encoder_outputs,EOS_token,output_lang):
    """During evaluation, perform a beam search to obtain a translation which will be closer to a global optimum, compared
    to the greedy search"""
    hypotheses = {initial_decoder_input[:, 0]: 0}
    best = {}
    decoder_hiddens = {translation: initial_decoder_hidden for translation in hypotheses.keys()}

    for di in range(max_length):
        best_scores = -float('inf') * torch.ones([beam_size])
        propositions = {}
        for translation, score in hypotheses.items():
            propositions, best, decoder_hiddens, best_scores = get_k_best_translations(beam_size, best_scores, translation,
                                                                                       score, decoder, decoder_hiddens,
                                                                                       encoder_outputs,
                                                                                       propositions, best)
        hypotheses = propositions

    return decode_final_translation(hypotheses,EOS_token ,output_lang)





def get_k_best_translations(k, scores, translation, score, decoder, decoder_hiddens, encoder_outputs, propositions,
                            best):
    """returns the k words which are the most likely to come after a given translation
    To do so, we apply the decoder on the last word of the translation while using the hidden paramaters which was
    previously computed for the translation and that we recorded in the dictionary decoder_hiddens"""

    decoder_output, decoder_hidden, decoder_attention = decoder(translation[-1].squeeze().detach(),
                                                                decoder_hiddens[translation], encoder_outputs)
    topv, topi = decoder_output.data.topk(k)
    for i in range(k):
        if score + topv[0][i] > torch.min(scores):
            s_min = torch.argsort(scores)[0]
            scores[s_min] = score + topv[0][i]
            new_translation = torch.cat((translation, topi[:, i]), -1)
            decoder_hiddens[new_translation] = decoder_hidden
            propositions[new_translation] = score + topv[0][i]
            propositions = keep_only_the_k_best_propositions(best, s_min, propositions)
            best[str(s_min)] = new_translation
    return propositions, best, decoder_hiddens, scores


def keep_only_the_k_best_propositions(best, s_min, propositions):
    """Best tracks the translations which have the best scores for a given length. If no translation has been recorded
    yet, a new key is made. Otherwise, the new translation will replace the old one in place"""
    try:
        old_translation = best[str(s_min)]
        propositions.pop(old_translation)
    except KeyError:
        pass
    return propositions


def decode_final_translation(hypotheses,EOS_token ,output_lang):
    """Once we have the final k outputs, we keep only the best one and we parse it to the proper format"""
    best_final_score = -float('inf')
    decoded_words = []
    for translation, score in hypotheses.items():
        if score > best_final_score:
            best_final_score = score
            decoded_words = []
            for token in translation:
                if token.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(output_lang.index2word[token.item()])
    return decoded_words


# def beam_search_decoder(max_length, beam_size, decoder_input, decoder, decoder_hidden, encoder_outputs, EOS_token,
#                         output_lang):
#     """the condensed function of all the ones above if you prefer to have a more global overview"""
#     hypotheses = {decoder_input[:, 0]: 0}
#     index = {}
#     decoder_hiddens = {translation: decoder_hidden for translation in hypotheses.keys()}
#
#     for di in range(max_length):
#         scores = -float('inf') * torch.ones([beam_size])
#         propositions = {}
#         for translation, score in hypotheses.items():
#             decoder_output, decoder_hidden, decoder_attention = decoder(translation[-1].squeeze().detach(),
#                                                                         decoder_hiddens[translation], encoder_outputs)
#             topv, topi = decoder_output.data.topk(beam_size)
#             for i in range(beam_size):
#                 if score + topv[0][i] > torch.min(scores):
#                     s_min = torch.argsort(scores)[0]
#                     scores[s_min] = score + topv[0][i]
#                     new_translation = torch.cat((translation, topi[:, i]), -1)
#                     decoder_hiddens[new_translation] = decoder_hidden
#                     propositions[new_translation] = score + topv[0][i]
#                     try:
#                         old_translation = index[str(s_min)]
#                         propositions.pop(old_translation)
#                     except KeyError:
#                         pass
#                     index[str(s_min)] = new_translation
#         hypotheses = propositions
#
#     best = -float('inf')
#     decoded_words = []
#     for translation, probability in hypotheses.items():
#         if probability > best:
#             best = probability
#             decoded_words = []
#             for token in translation:
#                 if token.item() == EOS_token:
#                     decoded_words.append('<EOS>')
#                     break
#                 decoded_words.append(output_lang.index2word[token.item()])
#
#     return decoded_words







