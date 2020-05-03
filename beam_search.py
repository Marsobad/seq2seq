import torch


def beam_search(beam_size, decoder, initial_decoder_input, max_length, initial_decoder_hidden, encoder_outputs,
                EOS_token, output_lang, max_hypotheses=None):
    """During evaluation, perform a beam search to obtain a translation which will be closer to a global optimum,
    compared to greedy search """
    hypotheses = {initial_decoder_input[:, 0]: 0}
    completed_hypotheses = {}
    hidden_states = {translation: initial_decoder_hidden for translation in hypotheses.keys()}

    if max_hypotheses:
        if beam_size > max_hypotheses:
            raise Exception("Max hypotheses should be greater than the beam size") from ParameterError
        for di in range(max_length):
            if len(completed_hypotheses.keys()) < max_hypotheses:
                hypotheses, hidden_states, completed_hypotheses = get_k_best_translations(beam_size, hypotheses,
                                                                                          decoder,
                                                                                          hidden_states,
                                                                                          encoder_outputs,
                                                                                          completed_hypotheses,
                                                                                          EOS_token,
                                                                                          max_length)
            else:
                break
    else:
        for di in range(max_length):
            hypotheses, hidden_states, completed_hypotheses = get_k_best_translations(beam_size, hypotheses, decoder,
                                                                                      hidden_states,
                                                                                      encoder_outputs,
                                                                                      completed_hypotheses, EOS_token,
                                                                                      max_length)
    return decode_final_translation(completed_hypotheses, EOS_token, output_lang)


def get_k_best_translations(k, hypotheses, decoder, hidden_states, encoder_outputs, complete_hypotheses, EOS_token,
                            max_length):
    """returns the k words which are the most likely to come after a given translation
    To do so, we apply the decoder on the last word of the translation while using the hidden parameters which were
    previously computed for the sub translation and that we recorded in the dictionary named 'decoder_hiddens'"""
    propositions = {}
    scores_best_translations = {}
    scores = -float('inf') * torch.ones([k])

    for translation, score in hypotheses.items():
        decoder_output, decoder_hidden, decoder_attention = decoder(translation[-1].squeeze().detach(),
                                                                    hidden_states[translation], encoder_outputs)
        topv, topi = decoder_output.data.topk(k)
        for i in range(k):
            if score + topv[0][i] > torch.min(scores):
                s_min = torch.argsort(scores)[0]
                scores[s_min] = score + topv[0][i]
                new_translation = torch.cat((translation, topi[:, i]), -1)
                hidden_states[new_translation] = decoder_hidden
                propositions[new_translation] = score + topv[0][i]
                propositions, scores_best_translations = keep_only_the_k_best_propositions(scores_best_translations,
                                                                                           s_min, propositions,
                                                                                           new_translation)

    complete_hypotheses, propositions = length_penalty(propositions, complete_hypotheses, EOS_token, max_length)
    return propositions, hidden_states, complete_hypotheses


def keep_only_the_k_best_propositions(scores_best_translations, s_min, propositions, new_translation):
    """Scores_best_translations tracks the translations which have the best scores for a given length. If no
    translation has been recorded yet, a new key is created. Otherwise, the new translation will replace the old one
    in place """
    try:
        old_translation = scores_best_translations[str(s_min)]
        propositions.pop(old_translation)
    except KeyError:
        pass
    scores_best_translations[str(s_min)] = new_translation
    return propositions, scores_best_translations


def length_penalty(propositions, completed_hypotheses, EOS_token, max_length):
    """When a translation is finished, it is stored in the completed hypotheses dictionary and removed from the
    pending hypotheses (incomplete_hypotheses). The score of the completed hypotheses is weighted by the
    length of the translations to perform length penalty"""
    incomplete_hypotheses = propositions.copy()

    for translation, score in propositions.items():
        if translation[-1] == EOS_token or translation.size()[0] == max_length:
            completed_hypotheses[translation] = score / translation.size()[0]
            incomplete_hypotheses.pop(translation)
    return completed_hypotheses, incomplete_hypotheses


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


class ParameterError(Exception):
    """Raised when the maximum number of completed hypotheses is smaller than the beam size"""
    pass
