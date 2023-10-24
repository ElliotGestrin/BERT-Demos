# Bert Demos
Demos written using BERT language models, mainly those from the Swedish Royal Library.

Currently, there are plans for two demos, which are independent.

## Chinese Whispers
In Chinese Whispers, a part of a text is randomly masked, after which an `Electra-Generator` model fills in the masks. This is then iteratively repeated.
It's also possible to use it to fill in the blanks in a given text.

For details, see [its README](./Chinese%20Whispers/README.md)

## AI Maskning
In AI Maskning, a `BERT-NER` model identifies various individuals, places, etc. which can then be marked and masked out.

For details, see [its README](./AI%20Masking/README.md)
## Sem-Search
In Sem-Search you're able to search a collection of texts by sentence semantics, rather than exact matches. There is a GUI available for this.

For details, see [its README](./Sem-Search/README.md).
