import torch


def beam_search(beam_size, decoder, initial_decoder_input, max_length, initial_decoder_hidden, encoder_outputs,
                EOS_token, output_lang):
    """During evaluation, perform a beam search to obtain a translation which will be closer to a global optimum, compared
    to greedy search"""
    hypotheses = {initial_decoder_input[:, 0]: 0}
    decoder_hiddens = {translation: initial_decoder_hidden for translation in hypotheses.keys()}

    for di in range(max_length):
        hypotheses, decoder_hiddens = get_k_best_translations(beam_size, hypotheses, decoder,
                                                                decoder_hiddens,
                                                                encoder_outputs)

    return decode_final_translation(hypotheses, EOS_token, output_lang)


def get_k_best_translations(k, hypotheses, decoder, decoder_hiddens, encoder_outputs):
    """returns the k words which are the most likely to come after a given translation
    To do so, we apply the decoder on the last word of the translation while using the hidden parameters which were
    previously computed for the sub translation and that we recorded in the dictionary named 'decoder_hiddens'"""

    propositions = {}
    scores_best_translations = {}
    scores = -float('inf') * torch.ones([k])
    for translation, score in hypotheses.items():
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
                propositions = keep_only_the_k_best_propositions(scores_best_translations, s_min, propositions)
                scores_best_translations[str(s_min)] = new_translation

    return propositions, decoder_hiddens


def keep_only_the_k_best_propositions(best, s_min, propositions):
    """Best tracks the translations which have the best scores for a given length. If no translation has been recorded
    yet, a new key is created. Otherwise, the new translation will replace the old one in place"""
    try:
        old_translation = best[str(s_min)]
        propositions.pop(old_translation)
    except KeyError:
        pass
    return propositions


def decode_final_translation(hypotheses, EOS_token, output_lang):
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
