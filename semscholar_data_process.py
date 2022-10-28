import ast
import json
import numpy as np
import pprint
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

def clean_text(text):

    words = text.split()
    words = [word for word in words if not (word[0]=='[' and word[-1]==']')]

    text = ' '.join(words).lower()

    return text

def main():

    data_file = 'semscholar_data.jsonl'
    out_file = 'preprocessed_data.jsonl'
    abstract_out_file = 'preprocessed_abstracts.jsonl'
    embedding_size = 768

    paper_data = []
    with open(data_file, 'r') as rf:
        for line in rf:
            this_paper_data = json.loads(line)
            paper_data.append(this_paper_data)
    
    print(f'paper_data: {len(paper_data)}')
    papers = []
    abstracts = []
    for paper in paper_data:
        if paper['body_text'] and paper['abstract']:
            paper_sections = []
            for section_dict in paper['body_text']:
                paper_section = {
                    'party': str(section_dict['section']),
                    'text': str(clean_text(section_dict['text'])),
                }
                # section = clean_text(section_dict['section'] + ' ' + section_dict['text'])
                paper_sections.append(paper_section)
            papers.append(paper_sections)

            abstract = {
                'party': 'abstract',
                'text': str(clean_text(paper['abstract'][0]['text'])),
            }
            abstracts.append([abstract])
            # print(papers)

    # pprint.pprint(f'papers: {papers}')
    with open(out_file, 'w') as f:
        for paper in papers:
            a = json.dumps(paper)
            print(a, file=f)
        # print(f'body_text: {[section.keys() for section in paper["body_text"]]}')

    with open(abstract_out_file, 'w') as wf:
        for abstract in abstracts:
            print(json.dumps(abstract), file=wf)

    # paper_data[0]
    # paper_embeddings = []
    # for idx, paper in enumerate(paper_data):
    #     if paper['s2data'] is not None:
    #         paper_embeddings.append(paper['s2data']['embedding']['vector'])
    #     else:
    #         print(f'no embedding found for paper {idx}: {paper}')
    #         paper_embeddings.append(np.zeros((768)))
    # paper_embeddings = np.array(paper_embeddings)

    # print(f'paper_embeddings: {paper_embeddings.shape}')

    # distances = 1 - pairwise_distances(paper_embeddings, metric='cosine')
    # print(distances)
    # for v_distances in distances:
    #     v_ranks = v_distances.argsort()
    #     print(v_ranks)

if __name__=="__main__":
    main()