import json
import pickle

import attr

from mac import config


@attr.s
class Question:
    qn_text = attr.ib()
    qn_ixs = attr.ib()
    answer_ix = attr.ib()
    image_ix = attr.ib()


def get_preprocess_questions(
        clevr_fs, output_fs, split_name, words_ixs=None):
    if not output_fs.exists('{}-questions.pkl'.format(split_name)):
        _preprocess_questions(
            clevr_fs, output_fs, split_name, words_ixs)

    questions = []
    with output_fs.open('{}-questions.pkl'.format(split_name), 'rb') as f:
        try:
            while True:
                questions.append(pickle.load(f))
        except EOFError:
            pass

    new_words_ixs = questions.pop(-1)

    return questions, new_words_ixs


def _preprocess_questions(clevr_fs, output_fs, split_name, words_ixs=None):
    with clevr_fs.open(f'questions/CLEVR_{split_name}_questions.json') as f:
        question_dataset = json.load(f)['questions']

    if words_ixs is None:
        words_ixs = {}
    answers_ix = config.getconfig()['answer_mapping']

    tmp_name = '{}-questions-tmp.pkl'.format(split_name)
    with output_fs.open(tmp_name, 'wb') as f:
        for question in question_dataset:
            qn_text = question['question']
            answer_ix = answers_ix[question['answer']]
            image_ix = question['image_index']

            qn_text = qn_text.lower()
            qn_text = qn_text.replace('?', '').replace(';', '')
            qn_words = qn_text.split(' ')
            qn_ixs = []
            for w in qn_words:
                if w not in words_ixs:
                    words_ixs[w] = len(words_ixs) + 1
                qn_ixs.append(words_ixs[w])

            qn_object = Question(
                qn_text,
                qn_ixs,
                answer_ix,
                image_ix
            )

            pickle.dump(qn_object, f)
        pickle.dump(words_ixs, f)

    output_fs.move(tmp_name, '{}-questions.pkl'.format(split_name))
