from Vocabulary import Vocabulary

s = "我 在 到了 说了额 可："
vocab = Vocabulary(s.split(' '))
print(vocab.word_to_index)
vocab.add_word("别")
print(vocab.word_to_index)
print(vocab.remove_word("到了"))
print(vocab.word_to_index)

print(Vocabulary.from_serializable(vocab.to_serializable()).word_to_index)