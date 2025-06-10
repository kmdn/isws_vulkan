# Evaluation Dataset from "The Lives of the Artists"

This directory contains data used for the Entity Extraction experiments of Entity Recognition and Entity Linking on Giorgio Vasari's *Lives of The Artists*.

This directory contains 8 files:

1. `sentences.csv`: containing 54 paragraphs extracted from 27 chapters collected from *Lives of The Artists* ([Project Gutenberg Edition](https://www.gutenberg.org/ebooks/25326))
2. `chapters.txt`: text file with the title of the chapters (for further references)
3. `entities.csv`: containing named entities labelled with the 4 entity types from CONLL-2003 (`PER`, `ORG`, `LOC`, `MISC`) and linked to Wikidata and
4. `works.csv`: containing artwork references (mostly visual works without titles) linked to Wikidata
5. `motifs.csv`: containing iconographic subjects annotated as subjects (`SUBJ`) or predicated state (`STATE`) and linked to [ICONCLASS](www.iconclass.org)
6. `spans.csv`: expliciting relations between the annotation of subjects (`SUBJ`) and the annotation of further predicated states (`STATE`) in `motifs.csv`
7. `terms.csv` **WORK IN PROGRESS**: containing generic artistic concepts linked to [Getty's AAT](https://www.getty.edu/research/tools/vocabularies/aat/)
8. `dates.csv` **WORK IN PROGRESS**: containing annotated dates (1463) or time intervals (*after the Death of Julius II*) 


For further information related to the annotation, [Guidelines](https://doi.org/10.5281/zenodo.7991340) are available on \[1\]. All the data was annotated using [Inception](https://inception-project.github.io/).


## Acknowledgments

We would like to thank [Dr. Lisa Dieckmann](https://nfdi4culture.de/about-us/people/fdcb9926-ef37-400a-8a64-064d7d48f198.html) and [Hans Brandhorst](https://nl.linkedin.com/in/hans-brandhorst-b032591b) for contributing to this project respectively on the annotation of visual works and the resolution of motifs with ICONCLASS.


## References

\[1\] Santini, Cristian, Tan, Mary Ann, Bruns, Oleksandra, Tietz, Tabea, Posthumus, Etienne, & Sack, Harald. (2023). Guidelines for the Annotation of ExtrART: Evaluation Dataset for Entity Extraction from The Lives Of The Artists (1550). Zenodo. [https://doi.org/10.5281/zenodo.7991340](https://doi.org/10.5281/zenodo.7991340)


