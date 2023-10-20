# Domstolsverket-Demo
Demos written for the application to the Domstolverket's AI-masking project.

Currently, there are plans for two demos, which are independent.

## Chinese Whispers
In Chinese Whispers, a part of a text is randomly masked, after which an `Electra-Generator` model fills in the masks. This is then iteratively repeated.

For details, see [its README](./Chinese%20Whispers/README.md)

## AI Maskning
In AI Maskning, a `BERT-NER` model identifies various individuals, places, etc. which can then be marked and masked out.

For details, see [its README](./AI%20Masking/README.md)
